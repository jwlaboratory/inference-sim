#!/usr/bin/env python3
"""Adaptive partial prefill: warm only on idle workers and only as deep as useful."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from gpu import Node

BTB_BENCH = ROOT / "bite-the-bullet" / "synthetic_trace_benchmark.py"


def load_benchmark():
    spec = importlib.util.spec_from_file_location("btb_synthetic_trace_benchmark", BTB_BENCH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = load_benchmark()


@dataclass
class PendingWarm:
    ready: float
    node: Node
    blocks: list[str]
    key: tuple[str, ...]
    bytes: int
    duration: float


def ceil_blocks(tokens: int, block_tokens: int) -> int:
    return math.ceil(tokens / block_tokens)


def bench_args(args: argparse.Namespace, warm_blocks: int) -> argparse.Namespace:
    return argparse.Namespace(
        num_bursts=args.num_bursts,
        burst_size=args.burst_size,
        prefix_tokens=args.prefix_tokens,
        suffix_tokens=args.suffix_tokens,
        output_tokens=args.output_tokens,
        first_burst_s=args.first_burst_s,
        burst_spacing_s=args.burst_spacing_s,
        burst_window_s=args.burst_window_s,
        num_decoys=args.num_decoys,
        decoy_size=args.decoy_size,
        decoy_window_s=args.decoy_window_s,
        background_requests=args.background_requests,
        background_prefix_tokens=args.background_prefix_tokens,
        background_output_tokens=args.background_output_tokens,
        lead_s=args.lead_s,
        active_ttl_s=args.active_ttl_s,
        replicas=args.replicas,
        warm_blocks=warm_blocks,
        key_blocks=args.key_blocks,
        precision_sweep=args.precision_sweep,
        recall_sweep=[args.recall],
        trials=args.trials,
        rdma_gbps=args.rdma_gbps,
        hbm_only=args.hbm_only,
        block_tokens=args.block_tokens,
        model_preset=args.model_preset,
        num_replicas=args.num_replicas,
        gpus_per_replica=args.gpus_per_replica,
        gpu=args.gpu,
        imbalance_abs=args.imbalance_abs,
        imbalance_rel=args.imbalance_rel,
        max_batch=args.max_batch,
        seed=args.seed,
        out=args.out,
    )


def node_idle(node: Node, now: float) -> bool:
    return not node.running and not node.waiting and node.now <= now + 1e-12


def process_ready(
    pending: list[PendingWarm],
    now: float,
    planned: set[tuple[tuple[str, ...], str]],
) -> tuple[list[PendingWarm], int]:
    keep = []
    ready = 0
    for warm in pending:
        if warm.ready <= now + 1e-12:
            warm.node.insert(warm.blocks)
            planned.discard((warm.key, warm.node.name))
            ready += 1
        else:
            keep.append(warm)
    return keep, ready


def local_blocks(node: Node, blocks) -> int:
    return bench.local_blocks(node, list(blocks))


def choose_adaptive_node(req, nodes: list[Node], now: float, cfg, active_keys, key_blocks: int):
    key = bench.prefix_key(req, key_blocks)
    if key and active_keys.get(key, -1.0) >= now:
        candidates = [node for node in nodes if local_blocks(node, req.blocks) >= key_blocks]
        if candidates:
            return min(candidates, key=lambda node: (-local_blocks(node, req.blocks), bench.node_load(node, now)))
    return bench.choose_cache_aware(req, nodes, now, cfg)


def fit_blocks_for_worker(
    node: Node,
    now: float,
    *,
    cfg,
    deadline: float,
    max_blocks: int,
    min_blocks: int,
    reserve_s: float,
    hbm_cache_fraction: float,
) -> int:
    budget_s = max(0.0, deadline - now - reserve_s)
    if budget_s <= 0:
        return 0
    block_s = node.prefill_time(cfg.BLOCK_TOKENS)
    time_fit = int(budget_s / block_s) if block_s > 0 else max_blocks
    if hbm_cache_fraction <= 0:
        cache_limit = max_blocks
    else:
        cache_limit = max(0, int(node.budget_blocks * hbm_cache_fraction) - len(node.hbm))
    blocks = min(max_blocks, time_fit, cache_limit)
    return blocks if blocks >= min_blocks else 0


def schedule_adaptive(
    signal,
    nodes: list[Node],
    now: float,
    cfg,
    *,
    key_blocks: int,
    replicas: int,
    max_warm_blocks: int,
    confidence: float,
    confidence_scaled: bool,
    planned: set[tuple[tuple[str, ...], str]],
    lead_s: float,
    reserve_s: float,
    hbm_cache_fraction: float,
) -> tuple[list[PendingWarm], dict[str, float]]:
    key = bench.prefix_key(signal.blocks, key_blocks)
    if not key:
        return [], {"warm_bytes": 0, "warm_busy_s": 0.0, "warm_count": 0, "skipped_busy": 0}

    confidence_cap = max_warm_blocks
    if confidence_scaled:
        confidence_cap = max(key_blocks, int(math.ceil(max_warm_blocks * max(0.0, min(1.0, confidence)))))
    max_blocks = min(max_warm_blocks, confidence_cap, len(signal.blocks))
    existing = [node for node in nodes if local_blocks(node, signal.blocks) >= key_blocks]
    planned_count = sum(1 for node in nodes if (key, node.name) in planned)
    needed = max(0, replicas - len(existing) - planned_count)
    candidates = [node for node in nodes if node not in existing and (key, node.name) not in planned]
    candidates = sorted(candidates, key=lambda node: bench.node_load(node, now))

    pending = []
    warm_bytes = 0
    warm_busy_s = 0.0
    skipped_busy = 0
    deadline = now + lead_s

    for target in candidates:
        if len(pending) >= needed:
            break
        if not node_idle(target, now):
            skipped_busy += 1
            continue
        blocks_fit = fit_blocks_for_worker(
            target,
            now,
            cfg=cfg,
            deadline=deadline,
            max_blocks=max_blocks,
            min_blocks=key_blocks,
            reserve_s=reserve_s,
            hbm_cache_fraction=hbm_cache_fraction,
        )
        if blocks_fit <= 0:
            continue
        blocks = list(signal.blocks[:blocks_fit])
        duration = target.prefill_time(len(blocks) * cfg.BLOCK_TOKENS)
        target.now = now + duration
        target.busy += duration
        nbytes = len(blocks) * target.block_bytes
        pending.append(PendingWarm(target.now, target, blocks, key, nbytes, duration))
        planned.add((key, target.name))
        warm_bytes += nbytes
        warm_busy_s += duration

    return pending, {
        "warm_bytes": warm_bytes,
        "warm_busy_s": warm_busy_s,
        "warm_count": len(pending),
        "skipped_busy": skipped_busy,
    }


def run_adaptive(
    cfg,
    requests,
    signals,
    signal_stats: dict[str, float],
    args: argparse.Namespace,
    *,
    max_warm_blocks: int,
    confidence_scaled: bool,
) -> dict:
    nodes = [Node(name, spec, n_gpus, cfg) for name, spec, n_gpus in cfg.CLUSTER]
    events = []
    pending: list[PendingWarm] = []
    planned: set[tuple[tuple[str, ...], str]] = set()
    active_keys: dict[tuple[str, ...], float] = {}
    stats = {
        **signal_stats,
        "warm_bytes": 0,
        "warm_busy_s": 0.0,
        "warm_count": 0,
        "ready_warms": 0,
        "skipped_busy": 0,
    }

    timeline = []
    timeline.extend(("signal", signal.time, i, signal) for i, signal in enumerate(signals))
    timeline.extend(("request", req.arrival, req.id, req) for req in requests)
    timeline.sort(key=lambda row: (row[1], 0 if row[0] == "signal" else 1, row[2]))

    for kind, now, _, item in timeline:
        bench.advance_all(nodes, now, cfg, events)
        pending, ready = process_ready(pending, now, planned)
        stats["ready_warms"] += ready
        active_keys = {key: until for key, until in active_keys.items() if until >= now}

        if kind == "signal":
            signal = item
            key = bench.prefix_key(signal.blocks, args.key_blocks)
            if key:
                active_keys[key] = max(active_keys.get(key, now), now + args.active_ttl_s)
            new_pending, warm_stats = schedule_adaptive(
                signal,
                nodes,
                now,
                cfg,
                key_blocks=args.key_blocks,
                replicas=args.replicas,
                max_warm_blocks=max_warm_blocks,
                confidence=signal_stats["signal_precision"],
                confidence_scaled=confidence_scaled,
                planned=planned,
                lead_s=args.lead_s,
                reserve_s=args.reserve_s,
                hbm_cache_fraction=args.hbm_cache_fraction,
            )
            pending.extend(new_pending)
            stats["warm_bytes"] += warm_stats["warm_bytes"]
            stats["warm_busy_s"] += warm_stats["warm_busy_s"]
            stats["warm_count"] += warm_stats["warm_count"]
            stats["skipped_busy"] += warm_stats["skipped_busy"]
            continue

        req = item
        chosen = choose_adaptive_node(req, nodes, now, cfg, active_keys, args.key_blocks)
        chosen.waiting.append(req)

    while pending:
        next_ready = min(warm.ready for warm in pending)
        bench.advance_all(nodes, next_ready, cfg, events)
        pending, ready = process_ready(pending, next_ready, planned)
        stats["ready_warms"] += ready

    bench.advance_all(nodes, None, cfg, events)
    events.sort(key=lambda event: event["id"])
    row = bench.summarize(bench.Policy("adaptive_empty_prefill", "predictive"), nodes, events, stats)
    row["policy"] = "adaptive_empty_conf" if confidence_scaled else "adaptive_empty_full"
    row["max_warm_tokens"] = max_warm_blocks * cfg.BLOCK_TOKENS
    row["confidence_scaled"] = confidence_scaled
    row["skipped_busy"] = stats["skipped_busy"]
    return row


def average_rows(rows: list[dict]) -> dict:
    out = bench.average_rows(rows)
    out["policy"] = rows[0]["policy"]
    out["max_warm_tokens"] = rows[0]["max_warm_tokens"]
    out["confidence_scaled"] = rows[0]["confidence_scaled"]
    out["skipped_busy"] = statistics.fmean(row["skipped_busy"] for row in rows)
    return out


def fixed_row(cfg, requests, signals, signal_stats, args, warm_blocks: int, name: str) -> dict:
    row = bench.run_policy(
        bench.Policy(name, "predictive", action="fake_prefill", replicas=args.replicas),
        cfg,
        requests,
        signals,
        key_blocks=args.key_blocks,
        warm_block_count=warm_blocks,
        active_ttl_s=args.active_ttl_s,
        signal_stats=signal_stats,
    )
    row["policy"] = name
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-bursts", type=int, default=8)
    parser.add_argument("--burst-size", type=int, default=500)
    parser.add_argument("--prefix-tokens", type=int, default=65536)
    parser.add_argument("--suffix-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--first-burst-s", type=float, default=20.0)
    parser.add_argument("--burst-spacing-s", type=float, default=40.0)
    parser.add_argument("--burst-window-s", type=float, default=1.0)
    parser.add_argument("--num-decoys", type=int, default=120)
    parser.add_argument("--decoy-size", type=int, default=1)
    parser.add_argument("--decoy-window-s", type=float, default=0.1)
    parser.add_argument("--background-requests", type=int, default=0)
    parser.add_argument("--background-prefix-tokens", type=int, default=2048)
    parser.add_argument("--background-output-tokens", type=int, default=1)
    parser.add_argument("--lead-s", type=float, default=6.0)
    parser.add_argument("--active-ttl-s", type=float, default=20.0)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--max-warm-tokens", type=int, default=65536)
    parser.add_argument("--fixed-warm-tokens", type=int, nargs="+", default=[32768, 65536])
    parser.add_argument("--key-blocks", type=int, default=8)
    parser.add_argument("--precision-sweep", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--recall", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--reserve-s", type=float, default=0.25)
    parser.add_argument(
        "--hbm-cache-fraction",
        type=float,
        default=0.0,
        help="0 disables the cap and lets normal LRU eviction handle HBM pressure",
    )
    parser.add_argument("--rdma-gbps", type=float, default=0.0)
    parser.add_argument("--hbm-only", action="store_true", default=True)
    parser.add_argument("--block-tokens", type=int, default=config.BLOCK_TOKENS)
    parser.add_argument("--model-preset", choices=["default", "glm52-int4"], default="default")
    parser.add_argument("--num-replicas", type=int)
    parser.add_argument("--gpus-per-replica", type=int)
    parser.add_argument("--gpu", choices=["H100", "H200", "B200", "B300", "A100"], default="H100")
    parser.add_argument("--imbalance-abs", type=int, default=config.IMBALANCE_ABS)
    parser.add_argument("--imbalance-rel", type=float, default=config.IMBALANCE_REL)
    parser.add_argument("--max-batch", type=int, default=config.MAX_BATCH)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="partial-prefill/adaptive_partial_prefill.json")
    return parser.parse_args()


def print_rows(rows: list[dict], base: dict) -> None:
    print("\nAdaptive/fixed comparison, delta vs cache_aware_no_warm on burst requests")
    print(
        f"{'policy':<24} {'P/R':>9} {'meanTTFT':>9} {'dMean':>9} "
        f"{'p95TTFT':>9} {'dP95':>9} {'warmGB':>8} {'busy':>8} {'skipBusy':>8}"
    )
    for row in rows:
        burst = row["burst"]
        print(
            f"{row['policy']:<24} "
            f"{row['signal_precision']:.2f}/{row['signal_recall']:.2f} "
            f"{burst['mean_ttft']:>8.2f}s "
            f"{burst['mean_ttft'] - base['mean_ttft']:>+8.2f}s "
            f"{burst['p95_ttft']:>8.2f}s "
            f"{burst['p95_ttft'] - base['p95_ttft']:>+8.2f}s "
            f"{row['warm_gb']:>7.1f} "
            f"{row['warm_busy_s']:>7.1f}s "
            f"{row.get('skipped_busy', 0):>8.1f}"
        )


def main() -> None:
    args = parse_args()
    max_warm_blocks = min(
        ceil_blocks(args.prefix_tokens, args.block_tokens),
        ceil_blocks(args.max_warm_tokens, args.block_tokens),
    )
    cfg = bench.make_cfg(bench_args(args, max_warm_blocks))
    jobs, requests = bench.build_jobs_and_requests(bench_args(args, max_warm_blocks), cfg)
    empty_stats = {
        "tp_signals": 0,
        "fp_signals": 0,
        "signals": 0,
        "signal_precision": 0.0,
        "signal_recall": 0.0,
    }
    baselines = {
        policy.name: bench.run_policy(
            policy,
            cfg,
            requests,
            [],
            key_blocks=args.key_blocks,
            warm_block_count=max_warm_blocks,
            active_ttl_s=args.active_ttl_s,
            signal_stats=empty_stats,
        )
        for policy in bench.baseline_policies()
    }
    base = baselines["cache_aware_no_warm"]["burst"]

    print(
        "Adaptive partial prefill setup: "
        f"bursts={args.num_bursts}x{args.burst_size}, bg={args.background_requests}, "
        f"cache_abs={args.imbalance_abs}, max_warm={args.max_warm_tokens}, "
        f"hbm_fraction={args.hbm_cache_fraction}, trials={args.trials}",
        flush=True,
    )
    bench.print_baselines(baselines)

    rows = []
    trial_rows = []
    fixed_blocks = [
        min(ceil_blocks(args.prefix_tokens, args.block_tokens), ceil_blocks(tokens, args.block_tokens))
        for tokens in args.fixed_warm_tokens
    ]

    for precision in args.precision_sweep:
        cells: dict[str, list[dict]] = {"adaptive_empty_full": [], "adaptive_empty_conf": []}
        for tokens in args.fixed_warm_tokens:
            cells[f"fixed_{tokens}"] = []

        for trial in range(args.trials):
            signals, signal_stats = bench.build_signals(
                jobs,
                precision=precision,
                recall=args.recall,
                lead_s=args.lead_s,
                seed=args.seed + trial * 100000 + int(precision * 100),
            )
            for confidence_scaled in [False, True]:
                row = run_adaptive(
                    cfg,
                    requests,
                    signals,
                    signal_stats,
                    args,
                    max_warm_blocks=max_warm_blocks,
                    confidence_scaled=confidence_scaled,
                )
                row["requested_precision"] = precision
                row["requested_recall"] = args.recall
                row["trial"] = trial
                cells[row["policy"]].append(row)
                trial_rows.append(row)
            for tokens, blocks in zip(args.fixed_warm_tokens, fixed_blocks):
                row = fixed_row(cfg, requests, signals, signal_stats, args, blocks, f"fixed_{tokens}")
                row["requested_precision"] = precision
                row["requested_recall"] = args.recall
                row["trial"] = trial
                cells[row["policy"]].append(row)
                trial_rows.append(row)

        for name, cell_rows in cells.items():
            if name.startswith("adaptive"):
                row = average_rows(cell_rows)
            else:
                row = bench.average_rows(cell_rows)
                row["policy"] = name
            row["requested_precision"] = precision
            row["requested_recall"] = args.recall
            rows.append(row)

    print_rows(rows, base)
    payload = {
        "args": vars(args),
        "baselines": baselines,
        "sweep": rows,
        "trial_rows": trial_rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
