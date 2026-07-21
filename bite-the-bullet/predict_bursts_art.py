#!/usr/bin/env python3
"""Train a small burst predictor on real ART-Chat rows.

Label definition:
  A request is positive if at least K later requests within H seconds share
  the same first N prefix-cache blocks. This is the concrete condition that
  would make proactive KV warming useful.

The model intentionally uses no sklearn dependency. It trains a standardized
logistic regression in pure Python and compares it with a simple momentum
rule such as "same prefix appeared recently >= threshold".
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import sys
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from workload import _rows, _total_rows


WINDOWS_S = (1.0, 5.0, 10.0, 30.0)
COLUMNS = [
    "request_id",
    "token_hash",
    "system_prompt_hash",
    "timestamp",
    "timestamp_ms",
    "input_length",
    "output_length",
    "hash_ids",
]
MODEL_FEATURES = [
    "input_tokens_log",
    "prefix_blocks_log",
    "message_bytes_log",
    "interarrival_s_log",
    "total_1s_log",
    "total_5s_log",
    "total_10s_log",
    "total_30s_log",
    "same_key_1s_log",
    "same_key_5s_log",
    "same_key_10s_log",
    "same_key_30s_log",
    "same_first_10s_log",
    "same_first_30s_log",
    "same_system_10s_log",
    "same_system_30s_log",
    "same_key_seen_log",
    "same_first_seen_log",
    "time_since_same_key_log",
    "time_since_same_first_log",
    "no_prior_same_key",
    "no_prior_same_first",
]


@dataclass
class Obs:
    window_id: int
    row_in_window: int
    request_id: str
    t: float
    input_tokens: int
    output_tokens: int
    blocks: list[str]
    key: tuple[str, ...]
    first_block: str
    system_hash: str
    message_bytes: int


@dataclass
class Example:
    window_id: int
    row_in_window: int
    request_id: str
    t: float
    label: int
    future_count: int
    features: dict[str, float]


def log1p_cap(x: float, cap: float | None = None) -> float:
    if cap is not None:
        x = min(x, cap)
    return math.log1p(max(0.0, x))


def key_for(blocks: list[str], key_blocks: int) -> tuple[str, ...]:
    if len(blocks) < key_blocks:
        return ()
    return tuple(blocks[:key_blocks])


def normalize_rows(rows: list[dict], window_id: int, key_blocks: int) -> list[Obs]:
    rows = sorted(rows, key=lambda row: row.get("timestamp_ms", row.get("timestamp", 0)))
    if not rows:
        return []
    t0 = rows[0].get("timestamp_ms", rows[0].get("timestamp", 0))
    obs = []
    for i, row in enumerate(rows):
        blocks = [str(x) for x in list(row.get("hash_ids") or [])]
        first = blocks[0] if blocks else ""
        timestamp = row.get("timestamp_ms", row.get("timestamp", 0))
        obs.append(
            Obs(
                window_id=window_id,
                row_in_window=i,
                request_id=str(row.get("request_id", f"w{window_id}:{i}")),
                t=(int(timestamp) - t0) / 1000.0,
                input_tokens=max(1, int(row.get("input_length") or 1)),
                output_tokens=max(1, int(row.get("output_length") or 1)),
                blocks=blocks,
                key=key_for(blocks, key_blocks),
                first_block=first,
                system_hash=str(row.get("system_prompt_hash") or ""),
                message_bytes=len(row.get("messages") or ""),
            )
        )
    return obs


def label_window(obs: list[Obs], horizon_s: float, future_k: int) -> dict[int, tuple[int, int]]:
    by_key: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for item in obs:
        if item.key:
            by_key[item.key].append(item.t)

    labels = {}
    if not obs:
        return labels
    last_t = obs[-1].t
    for i, item in enumerate(obs):
        if not item.key or item.t + horizon_s > last_t:
            continue
        times = by_key[item.key]
        lo = bisect.bisect_right(times, item.t)
        hi = bisect.bisect_right(times, item.t + horizon_s)
        future_count = hi - lo
        labels[i] = (1 if future_count >= future_k else 0, future_count)
    return labels


def pop_old(q: deque[float], now: float, window_s: float) -> None:
    cutoff = now - window_s
    while q and q[0] < cutoff:
        q.popleft()


def recent_count(history: dict, key, now: float, window_s: float) -> int:
    if not key:
        return 0
    q = history[key]
    pop_old(q, now, window_s)
    return len(q)


def build_examples(obs: list[Obs], horizon_s: float, future_k: int) -> list[Example]:
    labels = label_window(obs, horizon_s, future_k)
    total_history = {w: deque() for w in WINDOWS_S}
    key_history = {w: defaultdict(deque) for w in WINDOWS_S}
    first_history = {w: defaultdict(deque) for w in WINDOWS_S}
    system_history = {w: defaultdict(deque) for w in WINDOWS_S}
    seen_key = defaultdict(int)
    seen_first = defaultdict(int)
    last_key: dict[tuple[str, ...], float] = {}
    last_first: dict[str, float] = {}
    prev_t: float | None = None

    examples = []
    for i, item in enumerate(obs):
        interarrival = 0.0 if prev_t is None else max(0.0, item.t - prev_t)

        feats: dict[str, float] = {
            "input_tokens_log": log1p_cap(item.input_tokens),
            "prefix_blocks_log": log1p_cap(len(item.blocks)),
            "message_bytes_log": log1p_cap(item.message_bytes),
            "interarrival_s_log": log1p_cap(interarrival, 60.0),
            "same_key_seen": float(seen_key[item.key]) if item.key else 0.0,
            "same_first_seen": float(seen_first[item.first_block]) if item.first_block else 0.0,
        }
        feats["same_key_seen_log"] = log1p_cap(feats["same_key_seen"])
        feats["same_first_seen_log"] = log1p_cap(feats["same_first_seen"])

        if item.key and item.key in last_key:
            feats["time_since_same_key_log"] = log1p_cap(item.t - last_key[item.key], 3600.0)
            feats["no_prior_same_key"] = 0.0
        else:
            feats["time_since_same_key_log"] = log1p_cap(3600.0)
            feats["no_prior_same_key"] = 1.0
        if item.first_block and item.first_block in last_first:
            feats["time_since_same_first_log"] = log1p_cap(item.t - last_first[item.first_block], 3600.0)
            feats["no_prior_same_first"] = 0.0
        else:
            feats["time_since_same_first_log"] = log1p_cap(3600.0)
            feats["no_prior_same_first"] = 1.0

        for w in WINDOWS_S:
            total_q = total_history[w]
            pop_old(total_q, item.t, w)
            total = len(total_q)
            same_key = recent_count(key_history[w], item.key, item.t, w)
            same_first = recent_count(first_history[w], item.first_block, item.t, w)
            same_system = recent_count(system_history[w], item.system_hash, item.t, w)
            suffix = f"{int(w)}s"
            feats[f"total_{suffix}"] = float(total)
            feats[f"total_{suffix}_log"] = log1p_cap(total)
            feats[f"same_key_{suffix}"] = float(same_key)
            feats[f"same_key_{suffix}_log"] = log1p_cap(same_key)
            feats[f"same_first_{suffix}"] = float(same_first)
            feats[f"same_first_{suffix}_log"] = log1p_cap(same_first)
            feats[f"same_system_{suffix}"] = float(same_system)
            feats[f"same_system_{suffix}_log"] = log1p_cap(same_system)

        if i in labels:
            label, future_count = labels[i]
            examples.append(
                Example(
                    window_id=item.window_id,
                    row_in_window=item.row_in_window,
                    request_id=item.request_id,
                    t=item.t,
                    label=label,
                    future_count=future_count,
                    features=feats,
                )
            )

        for w in WINDOWS_S:
            total_history[w].append(item.t)
            if item.key:
                key_history[w][item.key].append(item.t)
            if item.first_block:
                first_history[w][item.first_block].append(item.t)
            if item.system_hash:
                system_history[w][item.system_hash].append(item.t)
        if item.key:
            seen_key[item.key] += 1
            last_key[item.key] = item.t
        if item.first_block:
            seen_first[item.first_block] += 1
            last_first[item.first_block] = item.t
        prev_t = item.t

    return examples


def parquet_files(dataset: str, split: str, config_name: str | None = None) -> list[dict]:
    url = f"https://datasets-server.huggingface.co/parquet?dataset={urllib.parse.quote(dataset)}"
    if config_name:
        url += f"&config={urllib.parse.quote(config_name)}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    files = [entry for entry in data["parquet_files"] if entry["split"] == split]
    if config_name:
        files = [entry for entry in files if entry.get("config") == config_name]
    if not files:
        raise RuntimeError(f"no parquet files found for {dataset}/{split}")
    return files


def load_examples_from_parquet(args: argparse.Namespace) -> tuple[list[Example], list[str]]:
    import fsspec
    import pyarrow.parquet as pq

    rng = random.Random(args.seed)
    files = parquet_files(args.dataset, args.split, args.config_name)[: args.max_parquet_files]
    fs = fsspec.filesystem("https")

    candidates = []
    metadata = {}
    for file_idx, entry in enumerate(files):
        with fs.open(entry["url"], "rb", block_size=1 << 20) as fh:
            pf = pq.ParquetFile(fh)
            metadata[file_idx] = {
                "entry": entry,
                "num_row_groups": pf.metadata.num_row_groups,
                "row_group_rows": [
                    pf.metadata.row_group(rg).num_rows for rg in range(pf.metadata.num_row_groups)
                ],
            }
        for rg, nrows in enumerate(metadata[file_idx]["row_group_rows"]):
            if nrows >= args.rows_per_window:
                candidates.append((file_idx, rg, nrows))

    if not candidates:
        raise RuntimeError(
            f"only {len(candidates)} parquet row groups can supply {args.rows_per_window} rows"
        )
    if len(candidates) >= args.windows:
        chosen = rng.sample(candidates, args.windows)
    else:
        chosen = [rng.choice(candidates) for _ in range(args.windows)]

    all_examples = []
    ids = []
    for w, (file_idx, rg, nrows) in enumerate(chosen):
        entry = metadata[file_idx]["entry"]
        start = rng.randrange(0, nrows - args.rows_per_window + 1)
        with fs.open(entry["url"], "rb", block_size=1 << 20) as fh:
            pf = pq.ParquetFile(fh)
            available = set(pf.schema_arrow.names)
            columns = [name for name in COLUMNS if name in available]
            table = pf.read_row_group(rg, columns=columns)
        rows = table.slice(start, args.rows_per_window).to_pylist()
        obs = normalize_rows(rows, w, args.key_blocks)
        examples = build_examples(obs, args.horizon_s, args.future_k)
        positives = sum(ex.label for ex in examples)
        ident = f"{entry['filename']}:rg{rg}:start{start}"
        ids.append(ident)
        print(
            f"window {w:02d} {ident:<24} rows={len(rows):<5} "
            f"usable={len(examples):<5} pos={positives:<5} "
            f"span={obs[-1].t if obs else 0:.1f}s"
        )
        all_examples.extend(examples)
    return all_examples, ids


def load_examples_from_rows_api(args: argparse.Namespace) -> tuple[list[Example], list[int]]:
    total = _total_rows(args.dataset)
    rng = random.Random(args.seed)
    if args.offset >= 0:
        offsets = [args.offset + i * args.rows_per_window for i in range(args.windows)]
    else:
        max_offset = max(0, total - args.rows_per_window - 1)
        offsets = sorted(rng.sample(range(max_offset), args.windows))

    all_examples = []
    for w, offset in enumerate(offsets):
        rows = _rows(args.dataset, args.split, offset, args.rows_per_window)
        obs = normalize_rows(rows, w, args.key_blocks)
        examples = build_examples(obs, args.horizon_s, args.future_k)
        positives = sum(ex.label for ex in examples)
        print(
            f"window {w:02d} offset={offset:<8} rows={len(rows):<5} "
            f"usable={len(examples):<5} pos={positives:<5} "
            f"span={obs[-1].t if obs else 0:.1f}s"
        )
        all_examples.extend(examples)
    return all_examples, offsets


def load_examples(args: argparse.Namespace) -> tuple[list[Example], list[int] | list[str]]:
    if args.source == "parquet":
        return load_examples_from_parquet(args)
    return load_examples_from_rows_api(args)


def split_examples(examples: list[Example], train_windows: int) -> tuple[list[Example], list[Example]]:
    train = [ex for ex in examples if ex.window_id < train_windows]
    test = [ex for ex in examples if ex.window_id >= train_windows]
    return train, test


def standardizer(train: list[Example], feature_names: list[str]) -> tuple[list[float], list[float]]:
    means = []
    stds = []
    for name in feature_names:
        vals = [ex.features.get(name, 0.0) for ex in train]
        mean = sum(vals) / len(vals) if vals else 0.0
        var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0
        std = math.sqrt(var) or 1.0
        means.append(mean)
        stds.append(std)
    return means, stds


def vectorize(ex: Example, feature_names: list[str], means: list[float], stds: list[float]) -> list[float]:
    return [1.0] + [
        (ex.features.get(name, 0.0) - mean) / std
        for name, mean, std in zip(feature_names, means, stds)
    ]


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def train_logreg(
    train: list[Example],
    feature_names: list[str],
    means: list[float],
    stds: list[float],
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> list[float]:
    xs = [vectorize(ex, feature_names, means, stds) for ex in train]
    ys = [ex.label for ex in train]
    pos = sum(ys)
    neg = len(ys) - pos
    pos_weight = neg / pos if pos else 1.0
    weights = [0.0 for _ in range(len(feature_names) + 1)]

    for _ in range(epochs):
        grad = [0.0 for _ in weights]
        weight_sum = 0.0
        for x, y in zip(xs, ys):
            p = sigmoid(sum(w * v for w, v in zip(weights, x)))
            sample_w = pos_weight if y else 1.0
            err = sample_w * (p - y)
            weight_sum += sample_w
            for j, val in enumerate(x):
                grad[j] += err * val
        denom = max(1e-9, weight_sum)
        for j in range(len(weights)):
            grad[j] /= denom
            if j:
                grad[j] += l2 * weights[j]
            weights[j] -= lr * grad[j]
    return weights


def predict_many(
    examples: list[Example],
    feature_names: list[str],
    means: list[float],
    stds: list[float],
    weights: list[float],
) -> list[float]:
    probs = []
    for ex in examples:
        x = vectorize(ex, feature_names, means, stds)
        probs.append(sigmoid(sum(w * v for w, v in zip(weights, x))))
    return probs


def binary_metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, float]:
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y and p)
    fp = sum(1 for y, p in zip(labels, preds) if not y and p)
    fn = sum(1 for y, p in zip(labels, preds) if y and not p)
    tn = sum(1 for y, p in zip(labels, preds) if not y and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(labels),
        "base_rate": sum(labels) / len(labels) if labels else 0.0,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if not positives:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    tp = 0
    fp = 0
    last_recall = 0.0
    ap = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        group_pos = sum(y for _, y in ranked[i:j])
        group_neg = (j - i) - group_pos
        tp += group_pos
        fp += group_neg
        recall = tp / positives
        precision = tp / (tp + fp) if tp + fp else 0.0
        ap += precision * (recall - last_recall)
        last_recall = recall
        i = j
    return ap


def auroc(labels: list[int], scores: list[float]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return 0.0
    ranked = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum = 0.0
    i = 0
    while i < len(ranked):
        j = i + 1
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * sum(y for _, y in ranked[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def best_threshold(labels: list[int], scores: list[float]) -> tuple[float, dict[str, float]]:
    best = (0.5, binary_metrics(labels, scores, 0.5))
    for i in range(1001):
        threshold = i / 1000
        metrics = binary_metrics(labels, scores, threshold)
        if (metrics["f1"], metrics["precision"], metrics["recall"]) > (
            best[1]["f1"],
            best[1]["precision"],
            best[1]["recall"],
        ):
            best = (threshold, metrics)
    return best


def best_rule(train: list[Example]) -> tuple[str, float]:
    candidates = ["same_key_1s", "same_key_5s", "same_key_10s", "same_key_30s", "same_first_10s"]
    labels = [ex.label for ex in train]
    best_name = candidates[0]
    best_threshold_value = 1.0
    best_score = -1.0
    for name in candidates:
        max_count = int(max((ex.features.get(name, 0.0) for ex in train), default=0))
        for threshold in range(1, max_count + 1):
            scores = [1.0 if ex.features.get(name, 0.0) >= threshold else 0.0 for ex in train]
            f1 = binary_metrics(labels, scores, 0.5)["f1"]
            if f1 > best_score:
                best_name = name
                best_threshold_value = float(threshold)
                best_score = f1
    return best_name, best_threshold_value


def rule_scores(examples: list[Example], name: str, threshold: float) -> list[float]:
    return [1.0 if ex.features.get(name, 0.0) >= threshold else 0.0 for ex in examples]


def top_k_precision(labels: list[int], scores: list[float], frac: float) -> float:
    if not labels:
        return 0.0
    k = max(1, int(len(labels) * frac))
    top = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)[:k]
    return sum(y for _, y in top) / len(top)


def summarize_model(name: str, examples: list[Example], scores: list[float], threshold: float) -> dict[str, float]:
    labels = [ex.label for ex in examples]
    metrics = binary_metrics(labels, scores, threshold)
    metrics["ap"] = average_precision(labels, scores)
    metrics["auroc"] = auroc(labels, scores)
    metrics["p_at_1pct"] = top_k_precision(labels, scores, 0.01)
    metrics["p_at_5pct"] = top_k_precision(labels, scores, 0.05)
    print(
        f"{name:<24} n={metrics['n']:<5} base={metrics['base_rate']:.1%} "
        f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f} "
        f"AP={metrics['ap']:.2f} AUROC={metrics['auroc']:.2f} "
        f"P@1%={metrics['p_at_1pct']:.2f} P@5%={metrics['p_at_5pct']:.2f}"
    )
    return metrics


def coefficient_table(feature_names: list[str], means: list[float], stds: list[float], weights: list[float]) -> list[dict]:
    rows = [{"feature": "bias", "weight": weights[0], "mean": 1.0, "std": 0.0}]
    for name, mean, std, weight in zip(feature_names, means, stds, weights[1:]):
        rows.append({"feature": name, "weight": weight, "mean": mean, "std": std})
    return sorted(rows, key=lambda row: abs(row["weight"]), reverse=True)


def save_model(
    path: Path,
    *,
    args: argparse.Namespace,
    offsets: list[int],
    feature_names: list[str],
    means: list[float],
    stds: list[float],
    weights: list[float],
    threshold: float,
    rule_name: str,
    rule_threshold: float,
    metrics: dict,
) -> None:
    payload = {
        "label": {
            "dataset": args.dataset,
            "config_name": args.config_name,
            "split": args.split,
            "key_blocks": args.key_blocks,
            "block_tokens": args.block_tokens,
            "horizon_s": args.horizon_s,
            "future_k": args.future_k,
            "definition": (
                "positive iff at least future_k later requests within horizon_s "
                "share the same first key_blocks hash_ids"
            ),
        },
        "data": {
            "offsets": offsets,
            "rows_per_window": args.rows_per_window,
            "train_windows": args.train_windows,
        },
        "model": {
            "type": "standardized_logistic_regression",
            "feature_names": feature_names,
            "means": means,
            "stds": stds,
            "weights": weights,
            "threshold": threshold,
        },
        "rule_baseline": {"feature": rule_name, "threshold": rule_threshold},
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=config.DATASET)
    parser.add_argument("--config-name")
    parser.add_argument("--split", default=config.DATASET_SPLIT)
    parser.add_argument("--source", choices=["parquet", "rows"], default="parquet")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--train-windows", type=int, default=6)
    parser.add_argument("--rows-per-window", type=int, default=800)
    parser.add_argument("--max-parquet-files", type=int, default=3)
    parser.add_argument("--offset", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--key-blocks", type=int, default=8)
    parser.add_argument("--block-tokens", type=int, default=config.BLOCK_TOKENS)
    parser.add_argument("--horizon-s", type=float, default=30.0)
    parser.add_argument("--future-k", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--model-out", default="bite-the-bullet/art_burst_model.json")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_windows <= 0 or args.train_windows >= args.windows:
        raise SystemExit("--train-windows must be between 1 and windows-1")

    print(
        "ART burst label: "
        f">={args.future_k} future requests in {args.horizon_s:g}s sharing first "
        f"{args.key_blocks} blocks ({args.key_blocks * config.BLOCK_TOKENS} tokens)."
    )
    examples, offsets = load_examples(args)
    train, test = split_examples(examples, args.train_windows)
    if not train or not test:
        raise SystemExit("not enough usable train/test examples; increase windows or rows-per-window")

    print(
        f"\ntrain={len(train)} examples from {args.train_windows} windows, "
        f"test={len(test)} examples from {args.windows - args.train_windows} windows"
    )
    print(
        f"train base={sum(ex.label for ex in train) / len(train):.1%}, "
        f"test base={sum(ex.label for ex in test) / len(test):.1%}"
    )

    means, stds = standardizer(train, MODEL_FEATURES)
    weights = train_logreg(
        train,
        MODEL_FEATURES,
        means,
        stds,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
    )

    train_scores = predict_many(train, MODEL_FEATURES, means, stds, weights)
    threshold, train_metrics = best_threshold([ex.label for ex in train], train_scores)
    test_scores = predict_many(test, MODEL_FEATURES, means, stds, weights)

    rule_name, rule_threshold = best_rule(train)
    train_rule = rule_scores(train, rule_name, rule_threshold)
    test_rule = rule_scores(test, rule_name, rule_threshold)

    print(f"\nthreshold chosen on train: model >= {threshold:.3f}")
    print(f"best train rule: {rule_name} >= {rule_threshold:g}")
    train_model_metrics = summarize_model("model train", train, train_scores, threshold)
    test_model_metrics = summarize_model("model test", test, test_scores, threshold)
    train_rule_metrics = summarize_model("rule train", train, train_rule, 0.5)
    test_rule_metrics = summarize_model("rule test", test, test_rule, 0.5)

    cold_test = [ex for ex in test if ex.features.get("same_key_seen", 0.0) == 0.0]
    if cold_test:
        cold_train = [ex for ex in train if ex.features.get("same_key_seen", 0.0) == 0.0]
        cold_scores = predict_many(cold_test, MODEL_FEATURES, means, stds, weights)
        summarize_model("model cold-start test", cold_test, cold_scores, threshold)
        if cold_train and sum(ex.label for ex in cold_train):
            cold_train_scores = predict_many(cold_train, MODEL_FEATURES, means, stds, weights)
            cold_threshold, _ = best_threshold([ex.label for ex in cold_train], cold_train_scores)
            summarize_model("cold-start tuned", cold_test, cold_scores, cold_threshold)
    early_test = [ex for ex in test if ex.features.get("same_key_30s", 0.0) <= 1.0]
    if early_test:
        early_train = [ex for ex in train if ex.features.get("same_key_30s", 0.0) <= 1.0]
        early_scores = predict_many(early_test, MODEL_FEATURES, means, stds, weights)
        summarize_model("model early test", early_test, early_scores, threshold)
        if early_train and sum(ex.label for ex in early_train):
            early_train_scores = predict_many(early_train, MODEL_FEATURES, means, stds, weights)
            early_threshold, _ = best_threshold([ex.label for ex in early_train], early_train_scores)
            summarize_model("early tuned", early_test, early_scores, early_threshold)

    print("\nTop model coefficients")
    for row in coefficient_table(MODEL_FEATURES, means, stds, weights)[:10]:
        print(f"{row['feature']:<26} {row['weight']:>8.3f}")

    metrics = {
        "train_model": train_model_metrics,
        "test_model": test_model_metrics,
        "train_rule": train_rule_metrics,
        "test_rule": test_rule_metrics,
    }
    if not args.no_save:
        out = Path(args.model_out)
        save_model(
            out,
            args=args,
            offsets=offsets,
            feature_names=MODEL_FEATURES,
            means=means,
            stds=stds,
            weights=weights,
            threshold=threshold,
            rule_name=rule_name,
            rule_threshold=rule_threshold,
            metrics=metrics,
        )
        print(f"\nsaved model: {out}")


if __name__ == "__main__":
    main()
