#!/usr/bin/env python3
"""Synthetic data-labeling spike benchmark for predictive KV warming.

The trace shape is intentionally different from ART/Mooncake replay:

  - A few real jobs each create a sudden fanout of hundreds of requests with
    the same long prefix.
  - Decoy jobs look similar to the predictor, but only create one/few requests.
  - The predictor quality is swept at the job-signal level: recall chooses how
    many real jobs are predicted, and precision controls false-positive decoys.

This lets us ask when "bite the bullet early" beats cache-aware routing under
the workload the idea is targeting, without depending on whether public traces
happen to contain large data-labeling bursts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from gpu import Node
from workload import Request

BTB_DIR = Path(__file__).resolve().parent


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


synthetic = load_local_module("btb_single_burst", BTB_DIR / "run.py")


@dataclass(frozen=True)
class Job:
    id: str
    kind: str
    start: float
    count: int
    window_s: float
    blocks: tuple[str, ...]


@dataclass(frozen=True)
class Signal:
    time: float
    job_id: str
    kind: str
    blocks: tuple[str, ...]


@dataclass(frozen=True)
class Policy:
    name: str
    route: str
    action: str | None = None
    replicas: int = 4


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


def make_cfg(args: argparse.Namespace) -> SimpleNamespace:
    cfg = synthetic.make_cfg(args.rdma_gbps)
    cfg.HBM_ONLY = args.hbm_only
    cfg.BLOCK_TOKENS = args.block_tokens
    cfg.IMBALANCE_ABS = args.imbalance_abs
    cfg.IMBALANCE_REL = args.imbalance_rel
    cfg.MAX_BATCH = args.max_batch
    return cfg


def make_blocks(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}:b{i}" for i in range(count))


def make_request(
    req_id: int,
    arrival: float,
    group: str,
    shared_blocks: tuple[str, ...],
    suffix_name: str,
    *,
    cfg: SimpleNamespace,
    suffix_tokens: int,
    output_tokens: int,
) -> Request:
    suffix_blocks = [f"{suffix_name}:s{i}" for i in range(ceil_blocks(suffix_tokens, cfg.BLOCK_TOKENS))]
    blocks = list(shared_blocks) + suffix_blocks
    input_tokens = len(blocks) * cfg.BLOCK_TOKENS
    extra = ceil_blocks(input_tokens + output_tokens, cfg.BLOCK_TOKENS) - len(blocks)
    cache_blocks = blocks + [f"{suffix_name}:o{i}" for i in range(max(0, extra))]
    return Request(
        id=req_id,
        arrival=arrival,
        group=group,
        prefix_tokens=input_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        blocks=blocks,
        cache_blocks=cache_blocks,
    )


def build_jobs_and_requests(args: argparse.Namespace, cfg: SimpleNamespace) -> tuple[list[Job], list[Request]]:
    rng = random.Random(args.seed)
    prefix_blocks = ceil_blocks(args.prefix_tokens, cfg.BLOCK_TOKENS)
    jobs: list[Job] = []
    requests: list[Request] = []
    req_id = 0

    for j in range(args.num_bursts):
        start = args.first_burst_s + j * args.burst_spacing_s
        blocks = make_blocks(f"burst:{j}", prefix_blocks)
        job = Job(f"burst:{j}", "burst", start, args.burst_size, args.burst_window_s, blocks)
        jobs.append(job)
        for i in range(args.burst_size):
            if args.burst_size == 1:
                arrival = start
            else:
                arrival = start + args.burst_window_s * i / (args.burst_size - 1)
            requests.append(
                make_request(
                    req_id,
                    arrival,
                    job.id,
                    blocks,
                    f"{job.id}:r{i}",
                    cfg=cfg,
                    suffix_tokens=args.suffix_tokens,
                    output_tokens=args.output_tokens,
                )
            )
            req_id += 1

    end_s = args.first_burst_s + max(1, args.num_bursts) * args.burst_spacing_s
    for j in range(args.num_decoys):
        start = rng.uniform(max(0.0, args.first_burst_s - args.lead_s), end_s)
        blocks = make_blocks(f"decoy:{j}", prefix_blocks)
        count = args.decoy_size
        job = Job(f"decoy:{j}", "decoy", start, count, args.decoy_window_s, blocks)
        jobs.append(job)
        for i in range(count):
            arrival = start if count == 1 else start + args.decoy_window_s * i / (count - 1)
            requests.append(
                make_request(
                    req_id,
                    arrival,
                    job.id,
                    blocks,
                    f"{job.id}:r{i}",
                    cfg=cfg,
                    suffix_tokens=args.suffix_tokens,
                    output_tokens=args.output_tokens,
                )
            )
            req_id += 1

    for i in range(args.background_requests):
        arrival = rng.uniform(0.0, end_s)
        blocks = make_blocks(f"bg:{i}", ceil_blocks(args.background_prefix_tokens, cfg.BLOCK_TOKENS))
        requests.append(
            make_request(
                req_id,
                arrival,
                f"bg:{i}",
                blocks,
                f"bg:{i}",
                cfg=cfg,
                suffix_tokens=args.suffix_tokens,
                output_tokens=args.background_output_tokens,
            )
        )
        req_id += 1

    requests.sort(key=lambda req: (req.arrival, req.id))
    for i, req in enumerate(requests):
        req.id = i
    return jobs, requests


def build_signals(
    jobs: list[Job],
    *,
    precision: float,
    recall: float,
    lead_s: float,
    seed: int,
) -> tuple[list[Signal], dict[str, float]]:
    rng = random.Random(seed)
    true_jobs = [job for job in jobs if job.kind == "burst"]
    decoys = [job for job in jobs if job.kind == "decoy"]
    rng.shuffle(true_jobs)
    rng.shuffle(decoys)

    tp_count = min(len(true_jobs), int(round(recall * len(true_jobs))))
    if tp_count == 0:
        fp_count = 0
    else:
        fp_count = int(round(tp_count * (1.0 / precision - 1.0))) if precision > 0 else len(decoys)
    fp_count = min(len(decoys), max(0, fp_count))

    selected_true = true_jobs[:tp_count]
    selected_false = decoys[:fp_count]
    signals = [
        Signal(max(0.0, job.start - lead_s), job.id, "tp", job.blocks)
        for job in selected_true
    ] + [
        Signal(max(0.0, job.start - lead_s), job.id, "fp", job.blocks)
        for job in selected_false
    ]
    signals.sort(key=lambda signal: (signal.time, signal.job_id))

    actual_precision = tp_count / len(signals) if signals else 0.0
    actual_recall = tp_count / len(true_jobs) if true_jobs else 0.0
    return signals, {
        "tp_signals": tp_count,
        "fp_signals": fp_count,
        "signals": len(signals),
        "signal_precision": actual_precision,
        "signal_recall": actual_recall,
    }


def prefix_key(req_or_blocks, key_blocks: int) -> tuple[str, ...]:
    blocks = req_or_blocks.blocks if hasattr(req_or_blocks, "blocks") else req_or_blocks
    if len(blocks) < key_blocks:
        return ()
    return tuple(blocks[:key_blocks])


def warm_blocks(req_or_blocks, blocks: int) -> list[str]:
    source = req_or_blocks.blocks if hasattr(req_or_blocks, "blocks") else req_or_blocks
    return list(source[: min(blocks, len(source))])


def local_blocks(node: Node, blocks: list[str] | tuple[str, ...]) -> int:
    return synthetic.local_blocks(node, list(blocks))


def full_nodes(nodes: list[Node], blocks: list[str]) -> list[Node]:
    if not blocks:
        return []
    want = len(blocks)
    return [node for node in nodes if local_blocks(node, blocks) >= want]


def node_load(node: Node, now: float) -> tuple[float, int, str]:
    return synthetic.load_key(node, now)


def choose_cache_aware(req: Request, nodes: list[Node], now: float, cfg: SimpleNamespace) -> Node:
    loads = [len(node.running) + len(node.waiting) for node in nodes]
    if max(loads) > cfg.IMBALANCE_ABS and max(loads) > cfg.IMBALANCE_REL * min(loads):
        return min(nodes, key=lambda node: node_load(node, now))
    return min(nodes, key=lambda node: (-local_blocks(node, req.blocks), node_load(node, now)))


def choose_seed_target(
    req: Request,
    nodes: list[Node],
    now: float,
    key_blocks: int,
    warm_block_count: int,
    replicas: int,
    planned: set[tuple[tuple[str, ...], str]],
) -> Node | None:
    key = prefix_key(req, key_blocks)
    blocks = warm_blocks(req, warm_block_count)
    if not key or not blocks:
        return None
    existing = full_nodes(nodes, blocks)
    planned_count = sum(1 for node in nodes if (key, node.name) in planned)
    if len(existing) + planned_count >= replicas:
        return None
    candidates = [node for node in nodes if node not in existing and (key, node.name) not in planned]
    if not candidates:
        return None
    return min(candidates, key=lambda node: node_load(node, now))


def choose_node(
    policy: Policy,
    req: Request,
    nodes: list[Node],
    now: float,
    cfg: SimpleNamespace,
    active_keys: dict[tuple[str, ...], float],
    key_blocks: int,
    warm_block_count: int,
) -> Node:
    if policy.route == "least_load":
        return min(nodes, key=lambda node: node_load(node, now))

    if policy.route == "pure_affinity":
        return min(nodes, key=lambda node: (-local_blocks(node, req.blocks), node_load(node, now)))

    if policy.route == "cache_aware":
        return choose_cache_aware(req, nodes, now, cfg)

    if policy.route == "predictive":
        key = prefix_key(req, key_blocks)
        if key and active_keys.get(key, -1.0) >= now:
            candidates = full_nodes(nodes, warm_blocks(req, warm_block_count))
            if candidates:
                return min(candidates, key=lambda node: node_load(node, now))
        return choose_cache_aware(req, nodes, now, cfg)

    raise ValueError(f"unknown route: {policy.route}")


def schedule_fake_prefill(
    signal: Signal,
    nodes: list[Node],
    now: float,
    cfg: SimpleNamespace,
    *,
    replicas: int,
    warm_block_count: int,
    planned: set[tuple[tuple[str, ...], str]],
) -> tuple[list[PendingWarm], dict[str, float]]:
    blocks = warm_blocks(signal.blocks, warm_block_count)
    key = prefix_key(blocks, len(blocks))
    if not key:
        return [], {"warm_bytes": 0, "warm_busy_s": 0.0, "warm_count": 0}

    existing = full_nodes(nodes, blocks)
    planned_count = sum(1 for node in nodes if (key, node.name) in planned)
    needed = max(0, replicas - len(existing) - planned_count)
    targets = [node for node in nodes if node not in existing and (key, node.name) not in planned]
    targets = sorted(targets, key=lambda node: node_load(node, now))[:needed]

    pending: list[PendingWarm] = []
    warm_bytes = 0
    warm_busy_s = 0.0
    tokens = len(blocks) * cfg.BLOCK_TOKENS
    for target in targets:
        start = max(target.now, now)
        duration = target.prefill_time(tokens)
        target.now = start + duration
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
    }


def process_ready_warms(
    pending: list[PendingWarm],
    now: float,
    planned: set[tuple[tuple[str, ...], str]],
) -> tuple[list[PendingWarm], int]:
    ready = 0
    keep = []
    for warm in pending:
        if warm.ready <= now + 1e-12:
            warm.node.insert(warm.blocks)
            planned.discard((warm.key, warm.node.name))
            ready += 1
        else:
            keep.append(warm)
    return keep, ready


def advance_all(
    nodes: list[Node],
    until: float | None,
    cfg: SimpleNamespace,
    events: list[dict],
) -> None:
    for node in nodes:
        synthetic.admit_and_decode_until(
            node,
            until,
            nodes,
            cfg,
            events,
            allow_remote=False,
        )


def run_policy(
    policy: Policy,
    cfg: SimpleNamespace,
    requests: list[Request],
    signals: list[Signal],
    *,
    key_blocks: int,
    warm_block_count: int,
    active_ttl_s: float,
    signal_stats: dict[str, float],
) -> dict:
    nodes = [Node(name, spec, n_gpus, cfg) for name, spec, n_gpus in cfg.CLUSTER]
    events: list[dict] = []
    pending: list[PendingWarm] = []
    planned: set[tuple[tuple[str, ...], str]] = set()
    active_keys: dict[tuple[str, ...], float] = {}
    stats = {
        **signal_stats,
        "warm_bytes": 0,
        "warm_busy_s": 0.0,
        "warm_count": 0,
        "ready_warms": 0,
    }

    timeline = []
    timeline.extend(("signal", signal.time, i, signal) for i, signal in enumerate(signals))
    timeline.extend(("request", req.arrival, req.id, req) for req in requests)
    timeline.sort(key=lambda row: (row[1], 0 if row[0] == "signal" else 1, row[2]))

    for kind, now, _, item in timeline:
        advance_all(nodes, now, cfg, events)
        pending, ready = process_ready_warms(pending, now, planned)
        stats["ready_warms"] += ready
        active_keys = {key: until for key, until in active_keys.items() if until >= now}

        if kind == "signal":
            signal = item
            key = prefix_key(signal.blocks, key_blocks)
            if key:
                active_keys[key] = max(active_keys.get(key, now), now + active_ttl_s)
            if policy.action == "fake_prefill":
                new_pending, warm_stats = schedule_fake_prefill(
                    signal,
                    nodes,
                    now,
                    cfg,
                    replicas=policy.replicas,
                    warm_block_count=warm_block_count,
                    planned=planned,
                )
                pending.extend(new_pending)
                stats["warm_bytes"] += warm_stats["warm_bytes"]
                stats["warm_busy_s"] += warm_stats["warm_busy_s"]
                stats["warm_count"] += warm_stats["warm_count"]
            continue

        req = item
        key = prefix_key(req, key_blocks)
        chosen = None
        if policy.action == "seed_real" and key and active_keys.get(key, -1.0) >= now:
            chosen = choose_seed_target(
                req,
                nodes,
                now,
                key_blocks,
                warm_block_count,
                policy.replicas,
                planned,
            )
            if chosen is not None:
                planned.add((key, chosen.name))
                stats["warm_bytes"] += len(warm_blocks(req, warm_block_count)) * chosen.block_bytes
                stats["warm_count"] += 1

        if chosen is None:
            chosen = choose_node(
                policy,
                req,
                nodes,
                now,
                cfg,
                active_keys,
                key_blocks,
                warm_block_count,
            )
        chosen.waiting.append(req)

    while pending:
        next_ready = min(warm.ready for warm in pending)
        advance_all(nodes, next_ready, cfg, events)
        pending, ready = process_ready_warms(pending, next_ready, planned)
        stats["ready_warms"] += ready

    advance_all(nodes, None, cfg, events)
    events.sort(key=lambda event: event["id"])
    return summarize(policy, nodes, events, stats)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[idx]


def metric_block(events: list[dict]) -> dict[str, float]:
    ttft = [event["start"] - event["arrival"] + event["reuse"] + event["prefill"] for event in events]
    queue = [event["start"] - event["arrival"] for event in events]
    lat = [event["finish"] - event["arrival"] for event in events]
    prefix_tok = sum(event["prefix_tokens"] for event in events)
    hit_tok = sum(event["hit"] for event in events)
    return {
        "requests": len(events),
        "mean_ttft": statistics.fmean(ttft) if ttft else 0.0,
        "p50_ttft": pct(ttft, 0.50),
        "p95_ttft": pct(ttft, 0.95),
        "p99_ttft": pct(ttft, 0.99),
        "max_ttft": max(ttft) if ttft else 0.0,
        "mean_queue": statistics.fmean(queue) if queue else 0.0,
        "p95_queue": pct(queue, 0.95),
        "mean_lat": statistics.fmean(lat) if lat else 0.0,
        "p95_lat": pct(lat, 0.95),
        "cache_hit": hit_tok / prefix_tok if prefix_tok else 0.0,
    }


def summarize(policy: Policy, nodes: list[Node], events: list[dict], stats: dict[str, float]) -> dict:
    burst_events = [event for event in events if event["group"].startswith("burst:")]
    decoy_events = [event for event in events if event["group"].startswith("decoy:")]
    span = max((event["finish"] for event in events), default=0.0)
    out_tok = sum(event["output_tokens"] for event in events)
    counts = {node.name: 0 for node in nodes}
    for event in burst_events:
        counts[event["node"]] += 1

    return {
        "policy": policy.name,
        "burst": metric_block(burst_events),
        "decoy": metric_block(decoy_events),
        "all": metric_block(events),
        "span": span,
        "throughput": out_tok / span if span else 0.0,
        "util": sum(node.busy for node in nodes) / (len(nodes) * span) if span else 0.0,
        "warm_gb": stats["warm_bytes"] / config.GB,
        "warm_busy_s": stats["warm_busy_s"],
        "warm_count": stats["warm_count"],
        "ready_warms": stats["ready_warms"],
        "signal_precision": stats["signal_precision"],
        "signal_recall": stats["signal_recall"],
        "tp_signals": stats["tp_signals"],
        "fp_signals": stats["fp_signals"],
        "signals": stats["signals"],
        "burst_node_counts": counts,
    }


def baseline_policies() -> list[Policy]:
    return [
        Policy("pure_cache_affinity", "pure_affinity"),
        Policy("cache_aware_no_warm", "cache_aware"),
        Policy("least_load_recompute", "least_load"),
    ]


def predictive_policies(replicas: int) -> list[Policy]:
    return [
        Policy("predict_fake_prefill", "predictive", action="fake_prefill", replicas=replicas),
        Policy("predict_seed_real", "predictive", action="seed_real", replicas=replicas),
    ]


def print_baselines(rows: dict[str, dict]) -> None:
    print("\nBaselines, burst requests only")
    print(f"{'policy':<24} {'meanTTFT':>9} {'p95TTFT':>9} {'meanLat':>9} {'hit':>6} {'split':>15}")
    for name in ["pure_cache_affinity", "cache_aware_no_warm", "least_load_recompute"]:
        row = rows[name]
        burst = row["burst"]
        split = "/".join(str(row["burst_node_counts"][name]) for name in sorted(row["burst_node_counts"]))
        print(
            f"{name:<24} {burst['mean_ttft']:>8.2f}s {burst['p95_ttft']:>8.2f}s "
            f"{burst['mean_lat']:>8.2f}s {burst['cache_hit']:>5.0%} {split:>15}"
        )


def print_sweep(rows: list[dict], baseline: dict) -> None:
    print("\nPredictor quality sweep, delta vs cache_aware_no_warm on burst requests")
    print(
        f"{'policy':<22} {'P/R':>9} {'meanTTFT':>9} {'dMean':>9} "
        f"{'p95TTFT':>9} {'dP95':>9} {'warmGB':>8} {'warmBusy':>9}"
    )
    base = baseline["burst"]
    for row in rows:
        burst = row["burst"]
        print(
            f"{row['policy']:<22} "
            f"{row['signal_precision']:.2f}/{row['signal_recall']:.2f} "
            f"{burst['mean_ttft']:>8.2f}s {burst['mean_ttft'] - base['mean_ttft']:>+8.2f}s "
            f"{burst['p95_ttft']:>8.2f}s {burst['p95_ttft'] - base['p95_ttft']:>+8.2f}s "
            f"{row['warm_gb']:>7.1f} {row['warm_busy_s']:>8.1f}s"
        )


def average_metric_dict(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    out = {}
    for key in keys:
        val = rows[0][key]
        if isinstance(val, (int, float)):
            out[key] = statistics.fmean(row[key] for row in rows)
    return out


def average_rows(rows: list[dict]) -> dict:
    first = rows[0]
    out = {
        "policy": first["policy"],
        "trials": len(rows),
        "burst": average_metric_dict([row["burst"] for row in rows]),
        "decoy": average_metric_dict([row["decoy"] for row in rows]),
        "all": average_metric_dict([row["all"] for row in rows]),
        "burst_node_counts": {
            name: statistics.fmean(row["burst_node_counts"][name] for row in rows)
            for name in first["burst_node_counts"]
        },
    }
    for key in [
        "span",
        "throughput",
        "util",
        "warm_gb",
        "warm_busy_s",
        "warm_count",
        "ready_warms",
        "signal_precision",
        "signal_recall",
        "tp_signals",
        "fp_signals",
        "signals",
    ]:
        out[key] = statistics.fmean(row[key] for row in rows)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-bursts", type=int, default=6)
    parser.add_argument("--burst-size", type=int, default=500)
    parser.add_argument("--prefix-tokens", type=int, default=65536)
    parser.add_argument("--suffix-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--first-burst-s", type=float, default=20.0)
    parser.add_argument("--burst-spacing-s", type=float, default=25.0)
    parser.add_argument("--burst-window-s", type=float, default=1.0)
    parser.add_argument("--num-decoys", type=int, default=60)
    parser.add_argument("--decoy-size", type=int, default=1)
    parser.add_argument("--decoy-window-s", type=float, default=0.1)
    parser.add_argument("--background-requests", type=int, default=240)
    parser.add_argument("--background-prefix-tokens", type=int, default=2048)
    parser.add_argument("--background-output-tokens", type=int, default=128)
    parser.add_argument("--lead-s", type=float, default=6.0)
    parser.add_argument("--active-ttl-s", type=float, default=20.0)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--warm-blocks", type=int, default=256)
    parser.add_argument("--key-blocks", type=int, default=8)
    parser.add_argument("--precision-sweep", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--recall-sweep", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--rdma-gbps", type=float, default=0.0)
    parser.add_argument("--hbm-only", action="store_true", default=True)
    parser.add_argument("--block-tokens", type=int, default=config.BLOCK_TOKENS)
    parser.add_argument("--imbalance-abs", type=int, default=64)
    parser.add_argument("--imbalance-rel", type=float, default=config.IMBALANCE_REL)
    parser.add_argument("--max-batch", type=int, default=config.MAX_BATCH)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="bite-the-bullet/synthetic_labeling_spike_sweep.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = make_cfg(args)
    jobs, requests = build_jobs_and_requests(args, cfg)
    print(
        "Synthetic labeling-spike benchmark: "
        f"{args.num_bursts} bursts x {args.burst_size} requests, "
        f"{args.prefix_tokens} shared prefix tokens, "
        f"{args.num_decoys} decoys, lead={args.lead_s:g}s, "
        f"cache_abs={args.imbalance_abs}, hbm_only={cfg.HBM_ONLY}",
        flush=True,
    )

    empty_signals, empty_stats = [], {
        "tp_signals": 0,
        "fp_signals": 0,
        "signals": 0,
        "signal_precision": 0.0,
        "signal_recall": 0.0,
    }
    baselines = {
        policy.name: run_policy(
            policy,
            cfg,
            requests,
            empty_signals,
            key_blocks=args.key_blocks,
            warm_block_count=args.warm_blocks,
            active_ttl_s=args.active_ttl_s,
            signal_stats=empty_stats,
        )
        for policy in baseline_policies()
    }
    print_baselines(baselines)

    sweep_rows = []
    trial_rows = []
    for recall in args.recall_sweep:
        for precision in args.precision_sweep:
            cell: dict[str, list[dict]] = {policy.name: [] for policy in predictive_policies(args.replicas)}
            for trial in range(args.trials):
                signals, signal_stats = build_signals(
                    jobs,
                    precision=precision,
                    recall=recall,
                    lead_s=args.lead_s,
                    seed=args.seed
                    + trial * 100000
                    + int(recall * 1000)
                    + int(precision * 100),
                )
                for policy in predictive_policies(args.replicas):
                    row = run_policy(
                        policy,
                        cfg,
                        requests,
                        signals,
                        key_blocks=args.key_blocks,
                        warm_block_count=args.warm_blocks,
                        active_ttl_s=args.active_ttl_s,
                        signal_stats=signal_stats,
                    )
                    row["requested_precision"] = precision
                    row["requested_recall"] = recall
                    row["trial"] = trial
                    cell[policy.name].append(row)
                    trial_rows.append(row)
            for policy in predictive_policies(args.replicas):
                row = average_rows(cell[policy.name])
                row["requested_precision"] = precision
                row["requested_recall"] = recall
                sweep_rows.append(row)
    print_sweep(sweep_rows, baselines["cache_aware_no_warm"])

    payload = {
        "args": vars(args),
        "jobs": {
            "bursts": sum(1 for job in jobs if job.kind == "burst"),
            "decoys": sum(1 for job in jobs if job.kind == "decoy"),
            "requests": len(requests),
        },
        "baselines": baselines,
        "sweep": sweep_rows,
        "trial_rows": trial_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
