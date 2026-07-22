#!/usr/bin/env python3
"""Run a compact ART-Chat BTB sweep across model and hardware regimes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "btb_utility_gate.py"
OUT_DIR = ROOT / "experiments" / "art_model_hardware_sweep"


BASE_ARGS = [
    "--dataset",
    "alessiotoniolo/ART-Chat-2.5M",
    "--config-name",
    "",
    "--block-tokens",
    "256",
    "--key-blocks",
    "8",
    "--warm-blocks",
    "8",
    "--rows-per-window",
    "500",
    "--windows",
    "6",
    "--train-windows",
    "3",
    "--horizon-s",
    "30",
    "--active-ttl-s",
    "30",
    "--max-candidates-per-window",
    "40",
    "--progress-every",
    "40",
    "--epochs",
    "400",
]

TOPK_ARGS = [
    "--gate-topk-options",
    "1",
    "2",
    "4",
    "8",
    "0",
    "--threshold-grid-step",
    "0.05",
    "--no-threshold-score-candidates",
]


SWEEP = [
    {
        "name": "70b_h100x4_base",
        "args": ["--model-preset", "default", "--gpu", "H100", "--gpus-per-replica", "4", "--num-replicas", "4"],
    },
    {
        "name": "70b_h100x4_fast_arrivals",
        "args": [
            "--model-preset",
            "default",
            "--gpu",
            "H100",
            "--gpus-per-replica",
            "4",
            "--num-replicas",
            "4",
            "--arrival-scale",
            "0.125",
            "--rows-per-window",
            "1000",
        ],
    },
    {
        "name": "70b_h100x4_slow_arrivals",
        "args": [
            "--model-preset",
            "default",
            "--gpu",
            "H100",
            "--gpus-per-replica",
            "4",
            "--num-replicas",
            "4",
            "--arrival-scale",
            "0.5",
        ],
    },
    {
        "name": "70b_a100x4_base",
        "args": ["--model-preset", "default", "--gpu", "A100", "--gpus-per-replica", "4", "--num-replicas", "4"],
    },
    {
        "name": "70b_h200x4_base",
        "args": ["--model-preset", "default", "--gpu", "H200", "--gpus-per-replica", "4", "--num-replicas", "4"],
    },
    {
        "name": "70b_b200x4_base",
        "args": ["--model-preset", "default", "--gpu", "B200", "--gpus-per-replica", "4", "--num-replicas", "4"],
    },
    {
        "name": "70b_h100x4_low_eff",
        "args": [
            "--model-preset",
            "default",
            "--gpu",
            "H100",
            "--gpus-per-replica",
            "4",
            "--num-replicas",
            "4",
            "--mfu",
            "0.35",
            "--mbu",
            "0.60",
        ],
    },
    {
        "name": "70b_h100x4_high_eff",
        "args": [
            "--model-preset",
            "default",
            "--gpu",
            "H100",
            "--gpus-per-replica",
            "4",
            "--num-replicas",
            "4",
            "--mfu",
            "0.65",
            "--mbu",
            "0.90",
        ],
    },
    {
        "name": "glm45_air_h100x4",
        "args": [
            "--model-preset",
            "glm45-air-int4",
            "--gpu",
            "H100",
            "--gpus-per-replica",
            "4",
            "--num-replicas",
            "4",
        ],
    },
    {
        "name": "glm45_h100x4",
        "args": ["--model-preset", "glm45-int4", "--gpu", "H100", "--gpus-per-replica", "4", "--num-replicas", "4"],
    },
    {
        "name": "glm52_h100x8",
        "args": ["--model-preset", "glm52-int4", "--gpu", "H100", "--gpus-per-replica", "8", "--num-replicas", "4"],
    },
    {
        "name": "kimi_k2_h100x8",
        "args": ["--model-preset", "kimi-k2-int4", "--gpu", "H100", "--gpus-per-replica", "8", "--num-replicas", "4"],
    },
    {
        "name": "kimi_k2_b200x4",
        "args": ["--model-preset", "kimi-k2-int4", "--gpu", "B200", "--gpus-per-replica", "4", "--num-replicas", "4"],
    },
    {
        "name": "kimi_k2_b300x4",
        "args": ["--model-preset", "kimi-k2-int4", "--gpu", "B300", "--gpus-per-replica", "4", "--num-replicas", "4"],
    },
    {
        "name": "dense1t_b300x4",
        "args": ["--model-preset", "dense1t-fp8", "--gpu", "B300", "--gpus-per-replica", "4", "--num-replicas", "4"],
    },
    {
        "name": "70b_h100x4_fast_topk",
        "args": [
            "--model-preset",
            "default",
            "--gpu",
            "H100",
            "--gpus-per-replica",
            "4",
            "--num-replicas",
            "4",
            "--arrival-scale",
            "0.125",
            "--rows-per-window",
            "1000",
            *TOPK_ARGS,
        ],
    },
    {
        "name": "70b_h100x4_slow_topk",
        "args": [
            "--model-preset",
            "default",
            "--gpu",
            "H100",
            "--gpus-per-replica",
            "4",
            "--num-replicas",
            "4",
            "--arrival-scale",
            "0.5",
            *TOPK_ARGS,
        ],
    },
    {
        "name": "70b_h200x4_topk",
        "args": [
            "--model-preset",
            "default",
            "--gpu",
            "H200",
            "--gpus-per-replica",
            "4",
            "--num-replicas",
            "4",
            *TOPK_ARGS,
        ],
    },
    {
        "name": "glm52_h100x8_topk",
        "args": [
            "--model-preset",
            "glm52-int4",
            "--gpu",
            "H100",
            "--gpus-per-replica",
            "8",
            "--num-replicas",
            "4",
            *TOPK_ARGS,
        ],
    },
    {
        "name": "kimi_k2_h100x8_topk",
        "args": [
            "--model-preset",
            "kimi-k2-int4",
            "--gpu",
            "H100",
            "--gpus-per-replica",
            "8",
            "--num-replicas",
            "4",
            *TOPK_ARGS,
        ],
    },
    {
        "name": "dense1t_b300x4_topk",
        "args": [
            "--model-preset",
            "dense1t-fp8",
            "--gpu",
            "B300",
            "--gpus-per-replica",
            "4",
            "--num-replicas",
            "4",
            *TOPK_ARGS,
        ],
    },
]


def pct(row: dict, base: dict, key: str) -> float:
    if not base.get(key):
        return 0.0
    return 100.0 * (float(row[key]) / float(base[key]) - 1.0)


def summarize_result(path: Path) -> dict:
    payload = json.loads(path.read_text())
    args = payload["args"]
    test = payload["test_summary"]
    base = test["baseline"]
    trained = test["trained_gate"]
    greedy = test["oracle_greedy"]
    return {
        "name": path.stem.removeprefix("btb_utility_gate_art_sweep_"),
        "model": args["model_preset"],
        "gpu": args["gpu"],
        "gpus_per_replica": args["gpus_per_replica"],
        "num_replicas": args["num_replicas"],
        "arrival_scale": args["arrival_scale"],
        "mfu": args.get("mfu", 0.0),
        "mbu": args.get("mbu", 0.0),
        "baseline_mean": base["mean_ttft"],
        "baseline_p95": base["p95_ttft"],
        "trained_mean_pct": pct(trained, base, "mean_ttft"),
        "trained_p95_pct": pct(trained, base, "p95_ttft"),
        "trained_triggers": trained["triggers"],
        "trained_warm_gb": trained["warm_gb"],
        "greedy_mean_pct": pct(greedy, base, "mean_ttft"),
        "greedy_p95_pct": pct(greedy, base, "p95_ttft"),
        "greedy_triggers": greedy["triggers"],
        "file": str(path.relative_to(ROOT)),
    }


def write_summary(rows: list[dict], path: Path) -> None:
    fields = [
        "name",
        "model",
        "gpu",
        "gpus_per_replica",
        "num_replicas",
        "arrival_scale",
        "mfu",
        "mbu",
        "baseline_mean",
        "baseline_p95",
        "trained_mean_pct",
        "trained_p95_pct",
        "trained_triggers",
        "trained_warm_gb",
        "greedy_mean_pct",
        "greedy_p95_pct",
        "greedy_triggers",
        "file",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row[field]) for field in fields))
    path.write_text("\n".join(lines) + "\n")


def print_markdown(rows: list[dict]) -> None:
    print("| Setup | Model | HW | Arrivals | Base mean | Trained dMean | Trained dP95 | Trig/win | Greedy dMean | Greedy dP95 |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        hw = f"{row['num_replicas']}x{row['gpus_per_replica']} {row['gpu']}"
        print(
            f"| {row['name']} | {row['model']} | {hw} | {row['arrival_scale']:g} | "
            f"{row['baseline_mean']:.3f}s | {row['trained_mean_pct']:+.1f}% | "
            f"{row['trained_p95_pct']:+.1f}% | {row['trained_triggers']:.1f} | "
            f"{row['greedy_mean_pct']:+.1f}% | {row['greedy_p95_pct']:+.1f}% |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="*", help="Run only these setup names.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [row for row in SWEEP if not args.only or row["name"] in set(args.only)]
    if args.only and len(selected) != len(set(args.only)):
        known = {row["name"] for row in SWEEP}
        missing = sorted(set(args.only) - known)
        raise SystemExit(f"unknown setup name(s): {', '.join(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for row in selected:
        out = args.out_dir / f"btb_utility_gate_art_sweep_{row['name']}.json"
        cmd = [sys.executable, str(EXPERIMENT), *BASE_ARGS, *row["args"], "--out", str(out)]
        print("$ " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        if out.exists() and not args.force:
            print(f"skip existing {out}", flush=True)
            continue
        try:
            subprocess.run(cmd, check=True, cwd=ROOT)
        except subprocess.CalledProcessError as exc:
            failed.append({"name": row["name"], "returncode": exc.returncode})
            print(f"FAILED {row['name']} with exit code {exc.returncode}", flush=True)

    if args.dry_run:
        return

    rows = [
        summarize_result(args.out_dir / f"btb_utility_gate_art_sweep_{row['name']}.json")
        for row in selected
        if (args.out_dir / f"btb_utility_gate_art_sweep_{row['name']}.json").exists()
    ]
    write_summary(rows, args.out_dir / "summary.tsv")
    if failed:
        (args.out_dir / "failures.json").write_text(json.dumps(failed, indent=2) + "\n")
    print_markdown(rows)
    if failed:
        print(f"\nFailed setups: {', '.join(item['name'] for item in failed)}")


if __name__ == "__main__":
    main()
