#!/usr/bin/env python3
"""Summarize BTB utility-gate result JSON files."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


def pct_delta(row: dict, base: dict, key: str) -> float:
    if not base.get(key):
        return 0.0
    return 100.0 * (float(row[key]) / float(base[key]) - 1.0)


def objective(row: dict, payload: dict) -> float:
    cfg = payload.get("objective") or {}
    metric = cfg.get("metric") or payload.get("args", {}).get("objective_metric", "mean_ttft")
    return (
        float(row.get(metric, 0.0))
        + float(cfg.get("warm_gb_cost", 0.0)) * float(row.get("warm_gb", 0.0))
        + float(cfg.get("warm_busy_cost", 0.0)) * float(row.get("warm_busy_s", 0.0))
        + float(cfg.get("trigger_cost", 0.0)) * float(row.get("triggers", 0.0))
    )


def objective_label(payload: dict) -> str:
    cfg = payload.get("objective") or {}
    args = payload.get("args", {})
    metric = cfg.get("metric") or args.get("objective_metric", "mean_ttft")
    parts = [metric]
    for name, label in [
        ("warm_gb_cost", "warm_gb"),
        ("warm_busy_cost", "warm_busy_s"),
        ("trigger_cost", "triggers"),
    ]:
        value = float(cfg.get(name, args.get(name, 0.0)) or 0.0)
        if value:
            parts.append(f"{value:g}*{label}")
    return "+".join(parts)


def label_for(path: Path, payload: dict) -> str:
    args = payload.get("args", {})
    source = args.get("source") or args.get("dataset") or path.stem
    jsonl_path = args.get("jsonl_path")
    if jsonl_path:
        source = Path(jsonl_path).stem
    return source


def row_for(path: Path) -> dict:
    payload = json.loads(path.read_text())
    test = payload["test_summary"]
    base = test["baseline"]
    trained = test["trained_gate"]
    greedy = test["oracle_greedy"]
    return {
        "file": str(path),
        "label": label_for(path, payload),
        "variant": objective_label(payload),
        "objective": objective(trained, payload) - objective(base, payload),
        "trained_mean_pct": pct_delta(trained, base, "mean_ttft"),
        "trained_p95_pct": pct_delta(trained, base, "p95_ttft"),
        "trained_triggers": trained["triggers"],
        "trained_warm_gb": trained["warm_gb"],
        "greedy_mean_pct": pct_delta(greedy, base, "mean_ttft"),
        "greedy_p95_pct": pct_delta(greedy, base, "p95_ttft"),
        "greedy_triggers": greedy["triggers"],
    }


def print_markdown(rows: list[dict]) -> None:
    print(
        "| Workload | dObjective | Trained dMean | Trained dP95 | "
        "Trig/win | Warm GB/win | Greedy dMean | Greedy dP95 |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            f"| {row['label']} | {row['objective']:+.4f} | "
            f"{row['trained_mean_pct']:+.1f}% | {row['trained_p95_pct']:+.1f}% | "
            f"{row['trained_triggers']:.1f} | {row['trained_warm_gb']:.3f} | "
            f"{row['greedy_mean_pct']:+.1f}% | {row['greedy_p95_pct']:+.1f}% |"
        )


def mean_stdev(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def grouped_rows(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        group_label = f"{row['label']} ({row['variant']})"
        groups.setdefault(group_label, []).append(row)

    out = []
    for label, items in sorted(groups.items()):
        row = {"label": label, "n": len(items)}
        for key in [
            "objective",
            "trained_mean_pct",
            "trained_p95_pct",
            "trained_triggers",
            "trained_warm_gb",
            "greedy_mean_pct",
            "greedy_p95_pct",
        ]:
            mean, stdev = mean_stdev([float(item[key]) for item in items])
            row[f"{key}_mean"] = mean
            row[f"{key}_stdev"] = stdev
        out.append(row)
    return out


def print_grouped_markdown(rows: list[dict]) -> None:
    print(
        "| Workload | N | dObjective | Trained dMean | Trained dP95 | "
        "Trig/win | Warm GB/win | Greedy dMean | Greedy dP95 |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in grouped_rows(rows):
        print(
            f"| {row['label']} | {row['n']} | "
            f"{row['objective_mean']:+.4f} +/- {row['objective_stdev']:.4f} | "
            f"{row['trained_mean_pct_mean']:+.1f}% +/- {row['trained_mean_pct_stdev']:.1f}% | "
            f"{row['trained_p95_pct_mean']:+.1f}% +/- {row['trained_p95_pct_stdev']:.1f}% | "
            f"{row['trained_triggers_mean']:.1f} +/- {row['trained_triggers_stdev']:.1f} | "
            f"{row['trained_warm_gb_mean']:.3f} +/- {row['trained_warm_gb_stdev']:.3f} | "
            f"{row['greedy_mean_pct_mean']:+.1f}% +/- {row['greedy_mean_pct_stdev']:.1f}% | "
            f"{row['greedy_p95_pct_mean']:+.1f}% +/- {row['greedy_p95_pct_stdev']:.1f}% |"
        )


def print_grouped_tsv(rows: list[dict]) -> None:
    fields = ["label", "n"]
    for key in [
        "objective",
        "trained_mean_pct",
        "trained_p95_pct",
        "trained_triggers",
        "trained_warm_gb",
        "greedy_mean_pct",
        "greedy_p95_pct",
    ]:
        fields.extend([f"{key}_mean", f"{key}_stdev"])
    print("\t".join(fields))
    for row in grouped_rows(rows):
        print("\t".join(str(row[field]) for field in fields))


def print_tsv(rows: list[dict]) -> None:
    fields = [
        "label",
        "objective",
        "trained_mean_pct",
        "trained_p95_pct",
        "trained_triggers",
        "trained_warm_gb",
        "greedy_mean_pct",
        "greedy_p95_pct",
        "file",
    ]
    print("\t".join(fields))
    for row in rows:
        print("\t".join(str(row[field]) for field in fields))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=["experiments/btb_utility_gate_*.json"])
    parser.add_argument("--format", choices=["markdown", "tsv"], default="markdown")
    parser.add_argument("--group-by-label", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = []
    for pattern in args.paths:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    rows = [row_for(Path(path)) for path in sorted(set(paths))]
    if args.group_by_label and args.format == "markdown":
        print_grouped_markdown(rows)
    elif args.group_by_label:
        print_grouped_tsv(rows)
    elif args.format == "markdown":
        print_markdown(rows)
    else:
        print_tsv(rows)


if __name__ == "__main__":
    main()
