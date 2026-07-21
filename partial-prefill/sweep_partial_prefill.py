#!/usr/bin/env python3
"""Sweep partial fake-prefill depth for data-labeling spike bursts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

BTB_BENCH = ROOT / "bite-the-bullet" / "synthetic_trace_benchmark.py"


def load_benchmark():
    spec = importlib.util.spec_from_file_location("btb_synthetic_trace_benchmark", BTB_BENCH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = load_benchmark()


def ceil_blocks(tokens: int, block_tokens: int) -> int:
    return math.ceil(tokens / block_tokens)


def metric_delta(row: dict, base: dict) -> dict:
    burst = row["burst"]
    return {
        "mean_ttft": burst["mean_ttft"] - base["mean_ttft"],
        "p95_ttft": burst["p95_ttft"] - base["p95_ttft"],
        "mean_lat": burst["mean_lat"] - base["mean_lat"],
        "p95_lat": burst["p95_lat"] - base["p95_lat"],
    }


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
    parser.add_argument("--key-blocks", type=int, default=8)
    parser.add_argument(
        "--warm-tokens",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384, 32768, 65536],
    )
    parser.add_argument("--precision-sweep", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--recall", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--rdma-gbps", type=float, default=0.0)
    parser.add_argument("--hbm-only", action="store_true", default=True)
    parser.add_argument("--block-tokens", type=int, default=config.BLOCK_TOKENS)
    parser.add_argument("--imbalance-abs", type=int, default=512)
    parser.add_argument("--imbalance-rel", type=float, default=config.IMBALANCE_REL)
    parser.add_argument("--max-batch", type=int, default=config.MAX_BATCH)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="partial-prefill/partial_prefill_sweep_abs512.json")
    return parser.parse_args()


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
        imbalance_abs=args.imbalance_abs,
        imbalance_rel=args.imbalance_rel,
        max_batch=args.max_batch,
        seed=args.seed,
        out=args.out,
    )


def run_cell(
    cfg,
    jobs,
    requests,
    args: argparse.Namespace,
    *,
    warm_tokens: int,
    warm_blocks: int,
    precision: float,
) -> dict:
    policy = bench.Policy(
        f"partial_fake_{warm_tokens}",
        "predictive",
        action="fake_prefill",
        replicas=args.replicas,
    )
    rows = []
    for trial in range(args.trials):
        signals, signal_stats = bench.build_signals(
            jobs,
            precision=precision,
            recall=args.recall,
            lead_s=args.lead_s,
            seed=args.seed + trial * 100000 + warm_blocks * 997 + int(precision * 100),
        )
        row = bench.run_policy(
            policy,
            cfg,
            requests,
            signals,
            key_blocks=args.key_blocks,
            warm_block_count=warm_blocks,
            active_ttl_s=args.active_ttl_s,
            signal_stats=signal_stats,
        )
        row["trial"] = trial
        rows.append(row)

    out = bench.average_rows(rows)
    out["policy"] = "partial_fake_prefill"
    out["requested_precision"] = precision
    out["requested_recall"] = args.recall
    out["warm_tokens"] = warm_tokens
    out["warm_blocks"] = warm_blocks
    out["warm_fraction"] = warm_tokens / args.prefix_tokens
    return out


def print_table(rows: list[dict], base: dict) -> None:
    print("\nPartial fake-prefill sweep, delta vs cache_aware_no_warm on burst requests")
    print(
        f"{'warm':>8} {'frac':>6} {'P/R':>9} {'meanTTFT':>9} {'dMean':>9} "
        f"{'p95TTFT':>9} {'dP95':>9} {'warmGB':>8} {'warmBusy':>9}"
    )
    for row in rows:
        burst = row["burst"]
        print(
            f"{row['warm_tokens']:>7} "
            f"{row['warm_fraction']:>5.0%} "
            f"{row['signal_precision']:.2f}/{row['signal_recall']:.2f} "
            f"{burst['mean_ttft']:>8.2f}s "
            f"{burst['mean_ttft'] - base['mean_ttft']:>+8.2f}s "
            f"{burst['p95_ttft']:>8.2f}s "
            f"{burst['p95_ttft'] - base['p95_ttft']:>+8.2f}s "
            f"{row['warm_gb']:>7.1f} "
            f"{row['warm_busy_s']:>8.1f}s"
        )


def frontier(rows: list[dict], base: dict) -> list[dict]:
    candidates = []
    for row in rows:
        delta = metric_delta(row, base)
        candidates.append(
            {
                "warm_tokens": row["warm_tokens"],
                "actual_precision": row["signal_precision"],
                "actual_recall": row["signal_recall"],
                "mean_ttft": row["burst"]["mean_ttft"],
                "p95_ttft": row["burst"]["p95_ttft"],
                "delta_mean_ttft": delta["mean_ttft"],
                "delta_p95_ttft": delta["p95_ttft"],
                "warm_gb": row["warm_gb"],
                "warm_busy_s": row["warm_busy_s"],
            }
        )
    return sorted(candidates, key=lambda row: (row["delta_p95_ttft"], row["warm_gb"]))[:8]


def main() -> None:
    args = parse_args()
    max_blocks = ceil_blocks(args.prefix_tokens, args.block_tokens)
    warm_depths = []
    for tokens in args.warm_tokens:
        blocks = min(max_blocks, ceil_blocks(tokens, args.block_tokens))
        actual_tokens = blocks * args.block_tokens
        if (actual_tokens, blocks) not in warm_depths:
            warm_depths.append((actual_tokens, blocks))

    cfg = bench.make_cfg(bench_args(args, warm_depths[-1][1]))
    jobs, requests = bench.build_jobs_and_requests(bench_args(args, warm_depths[-1][1]), cfg)
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
            warm_block_count=warm_depths[-1][1],
            active_ttl_s=args.active_ttl_s,
            signal_stats=empty_stats,
        )
        for policy in bench.baseline_policies()
    }
    base = baselines["cache_aware_no_warm"]["burst"]

    print(
        "Partial prefill setup: "
        f"{args.num_bursts} bursts x {args.burst_size}, "
        f"prefix={args.prefix_tokens} tokens, recall={args.recall:g}, "
        f"cache_abs={args.imbalance_abs}, trials={args.trials}",
        flush=True,
    )
    bench.print_baselines(baselines)

    rows = []
    for warm_tokens, warm_blocks in warm_depths:
        for precision in args.precision_sweep:
            rows.append(
                run_cell(
                    cfg,
                    jobs,
                    requests,
                    args,
                    warm_tokens=warm_tokens,
                    warm_blocks=warm_blocks,
                    precision=precision,
                )
            )

    print_table(rows, base)
    print("\nBest p95 frontier")
    for row in frontier(rows, base):
        print(
            f"warm={row['warm_tokens']:>6} P/R={row['actual_precision']:.2f}/{row['actual_recall']:.2f} "
            f"dMean={row['delta_mean_ttft']:+.2f}s dP95={row['delta_p95_ttft']:+.2f}s "
            f"warmGB={row['warm_gb']:.1f} warmBusy={row['warm_busy_s']:.1f}s"
        )

    payload = {
        "args": vars(args),
        "warm_depths": [{"tokens": tokens, "blocks": blocks} for tokens, blocks in warm_depths],
        "baselines": baselines,
        "sweep": rows,
        "frontier": frontier(rows, base),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
