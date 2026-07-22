#!/usr/bin/env python3
"""Train/test a utility gate for predictive prefix warming.

This is a direct test of the stronger BTB hypothesis:

  predict *when warming is useful*, not merely when same-prefix traffic is
  coming.

For each real-trace window, the script:

1. replays the baseline cache-aware/no-remote policy;
2. creates candidate warming decisions from observable request-prefix history;
3. labels candidates by counterfactual simulation: fire only this trigger and
   mark it positive if the selected metric improves;
4. trains a tiny standardized logistic regression on train-window candidates;
5. evaluates a trained gate and an oracle utility gate on held-out windows.

The experiment is intentionally small and dependency-light. It uses the repo's
Node model and a local copy of the BTB fake-prefill policy mechanics, because
the current main simulator does not expose speculative warming as a policy.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import random
import statistics
import sys
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from gpu import Node
from workload import Request


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
    "messages",
]
FEATURE_NAMES = [
    "input_tokens_log",
    "output_tokens_log",
    "prefix_blocks_log",
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
    "same_key_seen_log",
    "same_first_seen_log",
    "time_since_same_key_log",
    "time_since_same_first_log",
    "no_prior_same_key",
    "no_prior_same_first",
    "warm_tokens_log",
    "warm_seconds_log",
    "warm_seconds_per_future_horizon_log",
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
class Window:
    ident: str
    obs: list[Obs]
    requests: list[Request]
    features: list[dict[str, float]]
    future_counts: dict[int, int]
    candidates: list[int]


@dataclass
class Example:
    window_id: int
    request_id: int
    features: dict[str, float]
    utility: float
    label: int


@dataclass
class Seq:
    req: Request
    start: float
    reuse: float
    prefill: float
    hit: int
    tier: str
    left: int
    context: int


@dataclass
class PendingWarm:
    ready: float
    node: Node
    key: tuple[str, ...]
    blocks: list[str]
    bytes: int
    duration: float


def log1p_cap(value: float, cap: float | None = None) -> float:
    if cap is not None:
        value = min(value, cap)
    return math.log1p(max(0.0, value))


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[idx]


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def parquet_files(dataset: str, split: str, config_name: str | None) -> list[dict]:
    url = f"https://datasets-server.huggingface.co/parquet?dataset={urllib.parse.quote(dataset)}"
    if config_name:
        url += f"&config={urllib.parse.quote(config_name)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    files = [entry for entry in data["parquet_files"] if entry["split"] == split]
    if config_name:
        files = [entry for entry in files if entry.get("config") == config_name]
    if not files:
        raise RuntimeError(f"no parquet files found for {dataset}/{split}")
    return files


def key_for(blocks: list[str], key_blocks: int) -> tuple[str, ...]:
    if len(blocks) < key_blocks:
        return ()
    return tuple(blocks[:key_blocks])


def normalize_rows(rows: list[dict], window_id: int, key_blocks: int, arrival_scale: float) -> list[Obs]:
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
                t=((int(timestamp) - t0) / 1000.0) * arrival_scale,
                input_tokens=max(1, int(row.get("input_length") or 1)),
                output_tokens=max(1, int(row.get("output_length") or 1)),
                blocks=blocks,
                key=key_for(blocks, key_blocks),
                first_block=first,
                system_hash=str(row.get("system_prompt_hash") or ""),
                message_bytes=len(str(row.get("messages") or "")),
            )
        )
    return obs


def future_counts(obs: list[Obs], horizon_s: float) -> dict[int, int]:
    by_key: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for item in obs:
        if item.key:
            by_key[item.key].append(item.t)
    out = {}
    if not obs:
        return out
    last_t = obs[-1].t
    for i, item in enumerate(obs):
        if not item.key or item.t + horizon_s > last_t:
            continue
        times = by_key[item.key]
        lo = bisect.bisect_right(times, item.t)
        hi = bisect.bisect_right(times, item.t + horizon_s)
        out[i] = hi - lo
    return out


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


def build_feature_rows(
    obs: list[Obs],
    cfg: SimpleNamespace,
    warm_blocks_count: int,
    horizon_s: float,
) -> list[dict[str, float]]:
    total_history = {w: deque() for w in WINDOWS_S}
    key_history = {w: defaultdict(deque) for w in WINDOWS_S}
    first_history = {w: defaultdict(deque) for w in WINDOWS_S}
    seen_key = defaultdict(int)
    seen_first = defaultdict(int)
    last_key: dict[tuple[str, ...], float] = {}
    last_first: dict[str, float] = {}
    prev_t: float | None = None

    probe = Node("probe", cfg.CLUSTER[0][1], cfg.CLUSTER[0][2], cfg)
    warm_tokens = warm_blocks_count * cfg.BLOCK_TOKENS
    warm_s = probe.prefill_time(warm_tokens)

    rows = []
    for item in obs:
        interarrival = 0.0 if prev_t is None else max(0.0, item.t - prev_t)
        feats: dict[str, float] = {
            "input_tokens_log": log1p_cap(item.input_tokens),
            "output_tokens_log": log1p_cap(item.output_tokens),
            "prefix_blocks_log": log1p_cap(len(item.blocks)),
            "interarrival_s_log": log1p_cap(interarrival, 60.0),
            "same_key_seen": float(seen_key[item.key]) if item.key else 0.0,
            "same_first_seen": float(seen_first[item.first_block]) if item.first_block else 0.0,
            "warm_tokens_log": log1p_cap(warm_tokens),
            "warm_seconds_log": log1p_cap(warm_s),
            "warm_seconds_per_future_horizon_log": log1p_cap(warm_s / max(1e-9, horizon_s)),
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
            suffix = f"{int(w)}s"
            feats[f"total_{suffix}"] = float(total)
            feats[f"total_{suffix}_log"] = log1p_cap(total)
            feats[f"same_key_{suffix}"] = float(same_key)
            feats[f"same_key_{suffix}_log"] = log1p_cap(same_key)
            feats[f"same_first_{suffix}"] = float(same_first)
            feats[f"same_first_{suffix}_log"] = log1p_cap(same_first)

        rows.append(feats)

        for w in WINDOWS_S:
            total_history[w].append(item.t)
            if item.key:
                key_history[w][item.key].append(item.t)
            if item.first_block:
                first_history[w][item.first_block].append(item.t)
        if item.key:
            seen_key[item.key] += 1
            last_key[item.key] = item.t
        if item.first_block:
            seen_first[item.first_block] += 1
            last_first[item.first_block] = item.t
        prev_t = item.t
    return rows


def requests_from_obs(obs: list[Obs], cfg: SimpleNamespace) -> list[Request]:
    requests = []
    for i, item in enumerate(obs):
        blocks = list(item.blocks)
        extra = math.ceil((item.input_tokens + item.output_tokens) / cfg.BLOCK_TOKENS) - len(blocks)
        cache_blocks = blocks + [f"{item.request_id}#o{j}" for j in range(max(0, extra))]
        group = ":".join(item.key[:2]) if item.key else item.first_block[:12]
        requests.append(
            Request(
                id=i,
                arrival=item.t,
                group=group,
                prefix_tokens=min(len(blocks) * cfg.BLOCK_TOKENS, item.input_tokens),
                input_tokens=item.input_tokens,
                output_tokens=item.output_tokens,
                blocks=blocks,
                cache_blocks=cache_blocks,
            )
        )
    return requests


def make_cfg(args: argparse.Namespace) -> SimpleNamespace:
    cfg = SimpleNamespace(**config.as_dict())
    cfg.DISK_CACHE = False
    cfg.HBM_ONLY = args.hbm_only
    cfg.BLOCK_TOKENS = args.block_tokens
    cfg.MAX_BATCH = args.max_batch
    cfg.IMBALANCE_ABS = args.imbalance_abs
    cfg.IMBALANCE_REL = args.imbalance_rel
    if args.model_preset == "glm52-int4":
        cfg.PARAMS = 744e9
        cfg.ACTIVE_PARAMS = 40e9
        cfg.DTYPE_BYTES = 0.5
        cfg.LAYERS = 78
        cfg.KV_HEADS = 1
        cfg.HEAD_DIM = 288
    spec = getattr(config, args.gpu)
    spec = replace(spec, rdma_bw=args.rdma_gbps * config.GB)
    cfg.CLUSTER = [(f"node{i}", spec, args.gpus_per_replica) for i in range(args.num_replicas)]
    return cfg


def load_windows(args: argparse.Namespace, cfg: SimpleNamespace) -> list[Window]:
    if args.source == "jsonl":
        return load_jsonl_windows(args, cfg)
    if args.source == "burstgpt_csv":
        return load_burstgpt_windows(args, cfg)
    return load_mooncake_windows(args, cfg)


def load_mooncake_windows(args: argparse.Namespace, cfg: SimpleNamespace) -> list[Window]:
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
            row_groups = [pf.metadata.row_group(rg).num_rows for rg in range(pf.metadata.num_row_groups)]
        metadata[file_idx] = {"entry": entry, "row_groups": row_groups}
        for rg, nrows in enumerate(row_groups):
            if nrows >= args.rows_per_window:
                candidates.append((file_idx, rg, nrows))

    if not candidates:
        raise RuntimeError(f"no row groups can supply {args.rows_per_window} rows")

    rng.shuffle(candidates)
    if len(candidates) < args.windows:
        chosen = [rng.choice(candidates) for _ in range(args.windows)]
    else:
        chosen = candidates[: args.windows]

    windows = []
    for w, (file_idx, rg, nrows) in enumerate(chosen):
        entry = metadata[file_idx]["entry"]
        start = rng.randrange(0, nrows - args.rows_per_window + 1)
        with fs.open(entry["url"], "rb", block_size=1 << 20) as fh:
            pf = pq.ParquetFile(fh)
            available = set(pf.schema_arrow.names)
            columns = [name for name in COLUMNS if name in available]
            rows = pf.read_row_group(rg, columns=columns).slice(start, args.rows_per_window).to_pylist()

        obs = normalize_rows(rows, w, args.key_blocks, args.arrival_scale)
        requests = requests_from_obs(obs, cfg)
        features = build_feature_rows(obs, cfg, args.warm_blocks, args.horizon_s)
        fcounts = future_counts(obs, args.horizon_s)
        raw_candidates = [
            i
            for i, item in enumerate(obs)
            if item.key and i in fcounts and (args.include_cold_candidates or features[i].get("same_key_seen", 0.0) > 0)
        ]
        if args.max_candidates_per_window and len(raw_candidates) > args.max_candidates_per_window:
            raw_candidates = sorted(
                raw_candidates,
                key=lambda i: (
                    features[i].get("same_key_30s", 0.0),
                    features[i].get("same_key_seen", 0.0),
                    fcounts.get(i, 0),
                ),
                reverse=True,
            )[: args.max_candidates_per_window]
            raw_candidates.sort()

        ident = f"{entry['filename']}:rg{rg}:start{start}"
        positives = sum(1 for i in raw_candidates if fcounts.get(i, 0) >= args.future_k)
        print(
            f"window {w:02d} {ident:<24} rows={len(rows):<5} "
            f"span={obs[-1].t if obs else 0:.1f}s candidates={len(raw_candidates):<4} "
            f"future>=k={positives:<4}",
            flush=True,
        )
        windows.append(Window(ident, obs, requests, features, fcounts, raw_candidates))
    return windows


def load_burstgpt_rows(url: str, max_rows: int) -> list[dict]:
    rows = []
    with urllib.request.urlopen(url, timeout=120) as resp:
        text_lines = (line.decode("utf-8", errors="replace") for line in resp)
        reader = csv.DictReader(text_lines)
        for row in reader:
            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                break
    rows.sort(key=lambda row: float(row.get("Timestamp") or 0.0))
    return rows


def normalize_burstgpt_window(
    rows: list[dict],
    window_id: int,
    key_blocks: int,
    block_tokens: int,
    arrival_scale: float,
) -> list[Obs]:
    if not rows:
        return []
    t0 = float(rows[0].get("Timestamp") or 0.0)
    obs = []
    for i, row in enumerate(rows):
        request_id = f"burstgpt:{window_id}:{i}"
        session_id = str(row.get("Session ID") or "").strip()
        if session_id.lower() in {"", "nan", "none", "null"}:
            session_id = ""
        input_tokens = max(1, int(float(row.get("Request tokens") or 1)))
        output_tokens = max(1, int(float(row.get("Response tokens") or 1)))
        nblocks = max(1, math.ceil(input_tokens / block_tokens))
        if session_id:
            blocks = [f"burstgpt:session:{session_id}:b{j}" for j in range(nblocks)]
        else:
            blocks = [f"burstgpt:request:{request_id}:b{j}" for j in range(nblocks)]
        obs.append(
            Obs(
                window_id=window_id,
                row_in_window=i,
                request_id=request_id,
                t=(float(row.get("Timestamp") or 0.0) - t0) * arrival_scale,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                blocks=blocks,
                key=key_for(blocks, key_blocks),
                first_block=blocks[0] if blocks else "",
                system_hash=session_id or str(row.get("Model") or ""),
                message_bytes=0,
            )
        )
    return obs


def load_burstgpt_windows(args: argparse.Namespace, cfg: SimpleNamespace) -> list[Window]:
    rows = load_burstgpt_rows(args.burstgpt_url, args.burstgpt_max_rows)
    if len(rows) < args.rows_per_window:
        raise RuntimeError(f"only loaded {len(rows)} BurstGPT rows; need {args.rows_per_window}")

    rng = random.Random(args.seed)
    max_start = len(rows) - args.rows_per_window
    if args.burstgpt_starts:
        starts = [start for start in args.burstgpt_starts if 0 <= start <= max_start]
    elif args.burstgpt_sequential:
        starts = [i * args.rows_per_window for i in range(args.windows)]
        starts = [start for start in starts if start <= max_start]
    else:
        starts = sorted(rng.sample(range(max_start + 1), min(args.windows, max_start + 1)))
    if len(starts) < args.windows:
        raise RuntimeError(f"only found {len(starts)} BurstGPT windows; requested {args.windows}")

    windows = []
    for w, start in enumerate(starts[: args.windows]):
        rows_slice = rows[start : start + args.rows_per_window]
        obs = normalize_burstgpt_window(
            rows_slice,
            w,
            args.key_blocks,
            args.block_tokens,
            args.arrival_scale,
        )
        requests = requests_from_obs(obs, cfg)
        features = build_feature_rows(obs, cfg, args.warm_blocks, args.horizon_s)
        fcounts = future_counts(obs, args.horizon_s)
        raw_candidates = [
            i
            for i, item in enumerate(obs)
            if item.key and i in fcounts and (args.include_cold_candidates or features[i].get("same_key_seen", 0.0) > 0)
        ]
        if args.max_candidates_per_window and len(raw_candidates) > args.max_candidates_per_window:
            raw_candidates = sorted(
                raw_candidates,
                key=lambda i: (
                    features[i].get("same_key_30s", 0.0),
                    features[i].get("same_key_seen", 0.0),
                    fcounts.get(i, 0),
                ),
                reverse=True,
            )[: args.max_candidates_per_window]
            raw_candidates.sort()

        ident = f"BurstGPT_3.csv:start{start}"
        positives = sum(1 for i in raw_candidates if fcounts.get(i, 0) >= args.future_k)
        print(
            f"window {w:02d} {ident:<24} rows={len(rows_slice):<5} "
            f"span={obs[-1].t if obs else 0:.1f}s candidates={len(raw_candidates):<4} "
            f"future>=k={positives:<4}",
            flush=True,
        )
        windows.append(Window(ident, obs, requests, features, fcounts, raw_candidates))
    return windows


def load_jsonl_rows(path_or_url: str, max_rows: int) -> list[dict]:
    rows = []
    if path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(path_or_url, timeout=120) as resp:
            lines = (line.decode("utf-8", errors="replace") for line in resp)
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if max_rows and len(rows) >= max_rows:
                    break
        return rows
    with open(path_or_url, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def value_at(row: dict, field: str, default=None):
    cur = row
    for part in field.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def normalize_jsonl_window(
    rows: list[dict],
    window_id: int,
    args: argparse.Namespace,
) -> list[Obs]:
    if not rows:
        return []
    times = []
    elapsed = 0.0
    has_timestamp = value_at(rows[0], args.jsonl_timestamp_field) is not None
    for row in rows:
        if has_timestamp:
            times.append(float(value_at(row, args.jsonl_timestamp_field, 0.0)))
        else:
            times.append(elapsed)
            elapsed += float(value_at(row, args.jsonl_delay_field, 0.0)) / args.jsonl_delay_divisor
    t0 = times[0] if times else 0.0

    obs = []
    for i, (row, timestamp) in enumerate(zip(rows, times)):
        request_id = str(value_at(row, args.jsonl_request_id_field, f"jsonl:{window_id}:{i}"))
        input_tokens = max(1, int(float(value_at(row, args.jsonl_input_field, 1))))
        output_tokens = max(1, int(float(value_at(row, args.jsonl_output_field, 1))))
        raw_blocks = value_at(row, args.jsonl_hash_field)
        if raw_blocks:
            blocks = [str(block) for block in raw_blocks]
        else:
            session_id = str(value_at(row, args.jsonl_session_field, "") or "").strip()
            nblocks = max(1, math.ceil(input_tokens / args.block_tokens))
            if session_id:
                blocks = [f"jsonl:session:{session_id}:b{j}" for j in range(nblocks)]
            else:
                blocks = [f"jsonl:request:{request_id}:b{j}" for j in range(nblocks)]
        group = str(value_at(row, args.jsonl_group_field, "")) or str(value_at(row, args.jsonl_session_field, ""))
        obs.append(
            Obs(
                window_id=window_id,
                row_in_window=i,
                request_id=request_id,
                t=(timestamp - t0) * args.arrival_scale,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                blocks=blocks,
                key=key_for(blocks, args.key_blocks),
                first_block=blocks[0] if blocks else "",
                system_hash=group,
                message_bytes=len(str(value_at(row, args.jsonl_message_field, "") or "")),
            )
        )
    return obs


def load_jsonl_windows(args: argparse.Namespace, cfg: SimpleNamespace) -> list[Window]:
    rows = load_jsonl_rows(args.jsonl_path, args.jsonl_max_rows)
    if len(rows) < args.rows_per_window:
        raise RuntimeError(f"only loaded {len(rows)} JSONL rows; need {args.rows_per_window}")
    max_start = len(rows) - args.rows_per_window
    rng = random.Random(args.seed)
    if args.jsonl_starts:
        starts = [start for start in args.jsonl_starts if 0 <= start <= max_start]
    elif args.jsonl_sequential:
        starts = [i * args.rows_per_window for i in range(args.windows)]
        starts = [start for start in starts if start <= max_start]
    else:
        starts = sorted(rng.sample(range(max_start + 1), min(args.windows, max_start + 1)))
    if len(starts) < args.windows:
        raise RuntimeError(f"only found {len(starts)} JSONL windows; requested {args.windows}")

    windows = []
    for w, start in enumerate(starts[: args.windows]):
        rows_slice = rows[start : start + args.rows_per_window]
        obs = normalize_jsonl_window(rows_slice, w, args)
        requests = requests_from_obs(obs, cfg)
        features = build_feature_rows(obs, cfg, args.warm_blocks, args.horizon_s)
        fcounts = future_counts(obs, args.horizon_s)
        raw_candidates = [
            i
            for i, item in enumerate(obs)
            if item.key and i in fcounts and (args.include_cold_candidates or features[i].get("same_key_seen", 0.0) > 0)
        ]
        if args.max_candidates_per_window and len(raw_candidates) > args.max_candidates_per_window:
            raw_candidates = sorted(
                raw_candidates,
                key=lambda i: (
                    features[i].get("same_key_30s", 0.0),
                    features[i].get("same_key_seen", 0.0),
                    fcounts.get(i, 0),
                ),
                reverse=True,
            )[: args.max_candidates_per_window]
            raw_candidates.sort()

        ident = f"{args.jsonl_path}:start{start}"
        positives = sum(1 for i in raw_candidates if fcounts.get(i, 0) >= args.future_k)
        print(
            f"window {w:02d} {ident:<24} rows={len(rows_slice):<5} "
            f"span={obs[-1].t if obs else 0:.1f}s candidates={len(raw_candidates):<4} "
            f"future>=k={positives:<4}",
            flush=True,
        )
        windows.append(Window(ident, obs, requests, features, fcounts, raw_candidates))
    return windows


def local_blocks(node: Node, blocks: list[str]) -> int:
    return sum(node.match(blocks))


def load_key(node: Node, now: float) -> tuple[float, int, str]:
    busy_backlog = max(0.0, node.now - now)
    return (len(node.running) + len(node.waiting) + busy_backlog, len(node.waiting), node.name)


def prefix_source(req: Request, node: Node, nodes: list[Node], cfg: SimpleNamespace) -> tuple[int, float, str]:
    hbm_n, ram_n = node.match(req.blocks)
    if getattr(cfg, "HBM_ONLY", False):
        candidates = [(hbm_n, 0.0, "hbm")]
    else:
        candidates = [(hbm_n + ram_n, node.load_time(ram_n * node.block_bytes, "ram"), "hbm" if ram_n == 0 else "ram")]
    if cfg.RDMA_ON_ADMIT and node.tier_bw["rdma"] > 0:
        remote_n = max((sum(nd.match(req.blocks)) for nd in nodes if nd is not node), default=0)
        candidates.append((remote_n, node.load_time(remote_n * node.block_bytes, "rdma"), "rdma"))
    return max(candidates, key=lambda c: (c[0], -c[1]))


def finish_event(seq: Seq, node: Node) -> dict:
    req = seq.req
    return {
        "id": req.id,
        "arrival": req.arrival,
        "group": req.group,
        "prefix_tokens": req.prefix_tokens,
        "input_tokens": req.input_tokens,
        "output_tokens": req.output_tokens,
        "node": node.name,
        "start": seq.start,
        "finish": node.now,
        "reuse": seq.reuse,
        "prefill": seq.prefill,
        "decode": node.now - seq.start - seq.reuse - seq.prefill,
        "hit": seq.hit,
        "tier": seq.tier if seq.hit else "miss",
    }


def admit_and_decode_until(
    node: Node,
    until: float | None,
    nodes: list[Node],
    cfg: SimpleNamespace,
    events: list[dict],
) -> None:
    while True:
        while node.waiting and len(node.running) < cfg.MAX_BATCH:
            req = node.waiting[0]
            need = req.input_tokens + req.output_tokens
            used = sum(seq.req.input_tokens + seq.req.output_tokens for seq in node.running)
            if used + need > node.kv_budget:
                if not node.running:
                    raise ValueError(
                        f"request {req.id} needs {need} KV tokens but {node.name} has {node.kv_budget}"
                    )
                break

            node.waiting.popleft()
            n_blocks, load_s, tier = prefix_source(req, node, nodes, cfg)
            hit = min(n_blocks * cfg.BLOCK_TOKENS, req.prefix_tokens)
            if hit and load_s <= node.prefill_time(hit):
                reuse = load_s
                prefill = node.prefill_time(req.input_tokens - hit)
                used_hit = hit
                used_tier = tier
            else:
                reuse = 0.0
                prefill = node.prefill_time(req.input_tokens)
                used_hit = 0
                used_tier = "miss"

            node.running.append(Seq(req, node.now, reuse, prefill, used_hit, used_tier, req.output_tokens, req.input_tokens))
            node.now += reuse + prefill
            node.busy += reuse + prefill
            node.insert(req.cache_blocks)

        if until is not None and node.now >= until:
            return
        if not node.running:
            if until is not None:
                node.now = max(node.now, until)
            return

        batch = len(node.running)
        kv_tokens = sum(seq.context for seq in node.running)
        steps = min(seq.left for seq in node.running)
        if until is not None and node.decode_segment(steps, batch, kv_tokens) > until - node.now:
            lo, hi = 1, steps
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if node.decode_segment(mid, batch, kv_tokens) <= until - node.now:
                    lo = mid
                else:
                    hi = mid - 1
            steps = lo

        dur = node.decode_segment(steps, batch, kv_tokens)
        node.now += dur
        node.busy += dur
        for seq in node.running:
            seq.left -= steps
            seq.context += steps
        events.extend(finish_event(seq, node) for seq in node.running if seq.left == 0)
        node.running = [seq for seq in node.running if seq.left > 0]


def prefix_key(req: Request, key_blocks: int) -> tuple[str, ...]:
    if len(req.blocks) < key_blocks:
        return ()
    return tuple(req.blocks[:key_blocks])


def warm_prefix_blocks(req: Request, warm_blocks_count: int) -> list[str]:
    return list(req.blocks[: min(warm_blocks_count, len(req.blocks))])


def full_nodes(nodes: list[Node], blocks: list[str]) -> list[Node]:
    if not blocks:
        return []
    want = len(blocks)
    return [node for node in nodes if local_blocks(node, blocks) >= want]


def choose_cache_aware(req: Request, nodes: list[Node], now: float, cfg: SimpleNamespace) -> Node:
    loads = [len(node.running) + len(node.waiting) for node in nodes]
    if max(loads) > cfg.IMBALANCE_ABS and max(loads) > cfg.IMBALANCE_REL * min(loads):
        return min(nodes, key=lambda node: load_key(node, now))
    return min(nodes, key=lambda node: (-local_blocks(node, req.blocks), load_key(node, now)))


def choose_predictive(
    req: Request,
    nodes: list[Node],
    now: float,
    cfg: SimpleNamespace,
    key_blocks: int,
    warm_blocks_count: int,
    active_keys: dict[tuple[str, ...], float],
) -> Node:
    key = prefix_key(req, key_blocks)
    if key and active_keys.get(key, -1.0) >= now:
        candidates = full_nodes(nodes, warm_prefix_blocks(req, warm_blocks_count))
        if candidates:
            return min(candidates, key=lambda node: load_key(node, now))
    return choose_cache_aware(req, nodes, now, cfg)


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


def schedule_fake_prefill(
    req: Request,
    chosen: Node,
    nodes: list[Node],
    now: float,
    cfg: SimpleNamespace,
    key_blocks: int,
    warm_blocks_count: int,
    replicas: int,
    planned: set[tuple[tuple[str, ...], str]],
) -> tuple[list[PendingWarm], dict[str, float]]:
    key = prefix_key(req, key_blocks)
    blocks = warm_prefix_blocks(req, warm_blocks_count)
    if not key or not blocks:
        return [], {"warm_bytes": 0.0, "warm_busy_s": 0.0, "warm_count": 0.0}

    existing = full_nodes(nodes, blocks)
    planned_count = sum(1 for node in nodes if (key, node.name) in planned)
    implicit_chosen = 0 if chosen in existing else 1
    needed = max(0, replicas - len(existing) - planned_count - implicit_chosen)
    targets = [
        node
        for node in nodes
        if node is not chosen and node not in existing and (key, node.name) not in planned
    ]
    targets = sorted(targets, key=lambda node: load_key(node, now))[:needed]

    pending = []
    warm_bytes = 0
    warm_busy_s = 0.0
    tokens = len(blocks) * cfg.BLOCK_TOKENS
    for target in targets:
        start = max(target.now, now)
        duration = target.prefill_time(tokens)
        target.now = start + duration
        target.busy += duration
        nbytes = len(blocks) * target.block_bytes
        pending.append(PendingWarm(target.now, target, key, blocks, nbytes, duration))
        planned.add((key, target.name))
        warm_bytes += nbytes
        warm_busy_s += duration
    return pending, {"warm_bytes": warm_bytes, "warm_busy_s": warm_busy_s, "warm_count": len(pending)}


def run_policy(
    cfg: SimpleNamespace,
    window: Window,
    *,
    key_blocks: int,
    warm_blocks_count: int,
    replicas: int,
    active_ttl_s: float,
    trigger_ids: set[int] | None = None,
    score_by_id: dict[int, float] | None = None,
    threshold: float = 1.0,
) -> dict:
    nodes = [Node(name, spec, n_gpus, cfg) for name, spec, n_gpus in cfg.CLUSTER]
    events = []
    pending: list[PendingWarm] = []
    planned: set[tuple[tuple[str, ...], str]] = set()
    active_keys: dict[tuple[str, ...], float] = {}
    trigger_ids = trigger_ids or set()
    score_by_id = score_by_id or {}
    stats = {"triggers": 0, "warm_bytes": 0.0, "warm_busy_s": 0.0, "warm_count": 0.0, "ready_warms": 0}

    for req in window.requests:
        for node in nodes:
            admit_and_decode_until(node, req.arrival, nodes, cfg, events)
        pending, ready = process_ready_warms(pending, req.arrival, planned)
        stats["ready_warms"] += ready

        now = req.arrival
        active_keys = {key: until for key, until in active_keys.items() if until >= now}
        key = prefix_key(req, key_blocks)
        should_trigger = req.id in trigger_ids or score_by_id.get(req.id, -1.0) >= threshold
        if should_trigger and key:
            active_keys[key] = max(active_keys.get(key, now), now + active_ttl_s)
            stats["triggers"] += 1

        chosen = choose_predictive(req, nodes, now, cfg, key_blocks, warm_blocks_count, active_keys)
        chosen.waiting.append(req)

        if key and active_keys.get(key, -1.0) >= now:
            new_pending, warm_stats = schedule_fake_prefill(
                req,
                chosen,
                nodes,
                now,
                cfg,
                key_blocks,
                warm_blocks_count,
                replicas,
                planned,
            )
            pending.extend(new_pending)
            stats["warm_bytes"] += warm_stats["warm_bytes"]
            stats["warm_busy_s"] += warm_stats["warm_busy_s"]
            stats["warm_count"] += warm_stats["warm_count"]

    while pending:
        next_ready = min(warm.ready for warm in pending)
        for node in nodes:
            admit_and_decode_until(node, next_ready, nodes, cfg, events)
        pending, ready = process_ready_warms(pending, next_ready, planned)
        stats["ready_warms"] += ready

    for node in nodes:
        admit_and_decode_until(node, None, nodes, cfg, events)

    events.sort(key=lambda event: event["id"])
    return summarize(nodes, events, stats)


def summarize(nodes: list[Node], events: list[dict], stats: dict) -> dict:
    ttft = [event["start"] - event["arrival"] + event["reuse"] + event["prefill"] for event in events]
    queue = [event["start"] - event["arrival"] for event in events]
    lat = [event["finish"] - event["arrival"] for event in events]
    prefix_tok = sum(event["prefix_tokens"] for event in events)
    hit_tok = sum(event["hit"] for event in events)
    hbm_tok = sum(event["hit"] for event in events if event["tier"] == "hbm")
    out_tok = sum(event["output_tokens"] for event in events)
    span = max((event["finish"] for event in events), default=0.0)
    return {
        "requests": len(events),
        "span": span,
        "mean_ttft": statistics.fmean(ttft) if ttft else 0.0,
        "p95_ttft": pct(ttft, 0.95),
        "p99_ttft": pct(ttft, 0.99),
        "mean_queue": statistics.fmean(queue) if queue else 0.0,
        "p95_queue": pct(queue, 0.95),
        "mean_lat": statistics.fmean(lat) if lat else 0.0,
        "p95_lat": pct(lat, 0.95),
        "p99_lat": pct(lat, 0.99),
        "cache_hit": hit_tok / prefix_tok if prefix_tok else 0.0,
        "hbm_hit": hbm_tok / prefix_tok if prefix_tok else 0.0,
        "throughput": out_tok / span if span else 0.0,
        "util": sum(node.busy for node in nodes) / (len(nodes) * span) if span else 0.0,
        "critical_rdma_s": sum(event["reuse"] for event in events if event["tier"] == "rdma"),
        "triggers": stats["triggers"],
        "warm_gb": stats["warm_bytes"] / config.GB,
        "warm_busy_s": stats["warm_busy_s"],
        "warm_count": stats["warm_count"],
        "ready_warms": stats["ready_warms"],
    }


def objective(row: dict, args: argparse.Namespace) -> float:
    return (
        float(row[args.objective_metric])
        + args.warm_gb_cost * float(row.get("warm_gb", 0.0))
        + args.warm_busy_cost * float(row.get("warm_busy_s", 0.0))
        + args.trigger_cost * float(row.get("triggers", 0.0))
    )


def label_examples(args: argparse.Namespace, cfg: SimpleNamespace, windows: list[Window]) -> tuple[list[Example], dict]:
    examples: list[Example] = []
    baselines = {}
    for w, window in enumerate(windows):
        baseline = run_policy(
            cfg,
            window,
            key_blocks=args.key_blocks,
            warm_blocks_count=args.warm_blocks,
            replicas=args.replicas,
            active_ttl_s=args.active_ttl_s,
        )
        baselines[w] = baseline
        base_obj = objective(baseline, args)
        print(
            f"label window {w:02d}: baseline objective={base_obj:.6f}, "
            f"{args.objective_metric}={baseline[args.objective_metric]:.6f}, "
            f"candidates={len(window.candidates)}",
            flush=True,
        )
        for j, req_id in enumerate(window.candidates, start=1):
            warmed = run_policy(
                cfg,
                window,
                key_blocks=args.key_blocks,
                warm_blocks_count=args.warm_blocks,
                replicas=args.replicas,
                active_ttl_s=args.active_ttl_s,
                trigger_ids={req_id},
            )
            util = base_obj - objective(warmed, args)
            examples.append(
                Example(
                    window_id=w,
                    request_id=req_id,
                    features=window.features[req_id],
                    utility=util,
                    label=1 if util > args.min_utility else 0,
                )
            )
            if args.progress_every and j % args.progress_every == 0:
                pos = sum(ex.label for ex in examples if ex.window_id == w)
                print(f"  {j:4d}/{len(window.candidates)} labeled, positives={pos}", flush=True)
    return examples, baselines


def split_examples(examples: list[Example], train_windows: int) -> tuple[list[Example], list[Example]]:
    train = [ex for ex in examples if ex.window_id < train_windows]
    test = [ex for ex in examples if ex.window_id >= train_windows]
    return train, test


def window_slice_examples(examples: list[Example], start_window: int, end_window: int) -> list[Example]:
    return [ex for ex in examples if start_window <= ex.window_id < end_window]


def standardizer(train: list[Example]) -> tuple[list[float], list[float]]:
    means = []
    stds = []
    for name in FEATURE_NAMES:
        vals = [ex.features.get(name, 0.0) for ex in train]
        mean = sum(vals) / len(vals) if vals else 0.0
        var = sum((val - mean) ** 2 for val in vals) / len(vals) if vals else 0.0
        means.append(mean)
        stds.append(math.sqrt(var) or 1.0)
    return means, stds


def vectorize(ex: Example, means: list[float], stds: list[float]) -> list[float]:
    return [1.0] + [
        (ex.features.get(name, 0.0) - mean) / std
        for name, mean, std in zip(FEATURE_NAMES, means, stds)
    ]


def train_logreg(
    train: list[Example],
    means: list[float],
    stds: list[float],
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> list[float]:
    xs = [vectorize(ex, means, stds) for ex in train]
    ys = [ex.label for ex in train]
    pos = sum(ys)
    neg = len(ys) - pos
    pos_weight = neg / pos if pos else 1.0
    weights = [0.0 for _ in range(len(FEATURE_NAMES) + 1)]
    for _ in range(epochs):
        grad = [0.0 for _ in weights]
        denom = 0.0
        for x, y in zip(xs, ys):
            p = sigmoid(sum(w * val for w, val in zip(weights, x)))
            sample_w = pos_weight if y else 1.0
            err = sample_w * (p - y)
            denom += sample_w
            for j, val in enumerate(x):
                grad[j] += err * val
        denom = max(1e-9, denom)
        for j in range(len(weights)):
            grad[j] /= denom
            if j:
                grad[j] += l2 * weights[j]
            weights[j] -= lr * grad[j]
    return weights


def score_example(ex: Example, means: list[float], stds: list[float], weights: list[float]) -> float:
    x = vectorize(ex, means, stds)
    return sigmoid(sum(w * val for w, val in zip(weights, x)))


def select_score_ids(
    examples: list[Example],
    scores: dict[tuple[int, int], float],
    threshold: float,
    topk: int,
) -> set[int]:
    selected = [
        ex
        for ex in examples
        if scores[(ex.window_id, ex.request_id)] >= threshold
    ]
    selected.sort(key=lambda ex: scores[(ex.window_id, ex.request_id)], reverse=True)
    if topk > 0:
        selected = selected[:topk]
    return {ex.request_id for ex in selected}


def binary_metrics(
    examples: list[Example],
    scores: dict[tuple[int, int], float],
    threshold: float,
    topk: int = 0,
) -> dict:
    labels = [ex.label for ex in examples]
    selected_keys = set()
    if topk > 0:
        by_window: dict[int, list[Example]] = defaultdict(list)
        for ex in examples:
            by_window[ex.window_id].append(ex)
        for window_id, window_examples in by_window.items():
            selected_keys.update(
                (window_id, req_id)
                for req_id in select_score_ids(window_examples, scores, threshold, topk)
            )
        preds = [1 if (ex.window_id, ex.request_id) in selected_keys else 0 for ex in examples]
    else:
        preds = [1 if scores[(ex.window_id, ex.request_id)] >= threshold else 0 for ex in examples]
    tp = sum(1 for y, p in zip(labels, preds) if y and p)
    fp = sum(1 for y, p in zip(labels, preds) if not y and p)
    fn = sum(1 for y, p in zip(labels, preds) if y and not p)
    tn = sum(1 for y, p in zip(labels, preds) if not y and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(examples),
        "base_rate": sum(labels) / len(labels) if labels else 0.0,
        "threshold": threshold,
        "topk": topk,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def choose_threshold(train: list[Example], scores: dict[tuple[int, int], float]) -> tuple[float, dict]:
    best = (1.1, binary_metrics(train, scores, 1.1))
    for i in range(1001):
        threshold = i / 1000
        metrics = binary_metrics(train, scores, threshold)
        key = (metrics["f1"], metrics["precision"], metrics["recall"])
        best_key = (best[1]["f1"], best[1]["precision"], best[1]["recall"])
        if key > best_key:
            best = (threshold, metrics)
    return best


def choose_threshold_by_replay(
    args: argparse.Namespace,
    cfg: SimpleNamespace,
    windows: list[Window],
    examples: list[Example],
    scores: dict[tuple[int, int], float],
    baselines: dict,
) -> tuple[float, dict, dict]:
    by_window_examples: dict[int, list[Example]] = defaultdict(list)
    for ex in examples:
        by_window_examples[ex.window_id].append(ex)

    thresholds = {1.1}
    if args.threshold_grid_step > 0:
        steps = math.ceil(1.0 / args.threshold_grid_step)
        for i in range(steps + 1):
            thresholds.add(min(1.0, round(i * args.threshold_grid_step, 4)))
    if args.threshold_score_candidates:
        for ex in examples:
            thresholds.add(round(scores[(ex.window_id, ex.request_id)], 4))

    topk_options = sorted(set(args.gate_topk_options or [0]))
    best_threshold = 1.1
    best_topk = 0
    best_summary = None
    best_obj = math.inf
    best_warm = math.inf
    sweep = {}
    for threshold in sorted(thresholds):
        for topk in topk_options:
            rows = []
            for window in windows:
                window_id = window.obs[0].window_id
                trigger_ids = select_score_ids(
                    by_window_examples[window_id],
                    scores,
                    threshold,
                    topk,
                )
                rows.append(
                    run_policy(
                        cfg,
                        window,
                        key_blocks=args.key_blocks,
                        warm_blocks_count=args.warm_blocks,
                        replicas=args.replicas,
                        active_ttl_s=args.active_ttl_s,
                        trigger_ids=trigger_ids,
                    )
                )
            summary = aggregate(rows)
            obj = objective(summary, args)
            warm = summary["warm_busy_s"] + summary["warm_gb"]
            sweep[f"{threshold:.4f}:topk={topk or 'all'}"] = summary
            if (obj, warm) < (best_obj, best_warm):
                best_threshold = threshold
                best_topk = topk
                best_summary = summary
                best_obj = obj
                best_warm = warm

    assert best_summary is not None
    baseline_summary = aggregate([baselines[window.obs[0].window_id] for window in windows])
    return best_threshold, best_topk, best_summary, {
        "baseline": baseline_summary,
        "best": best_summary,
        "sweep": sweep,
    }


def aggregate(rows: list[dict]) -> dict:
    keys = [
        "mean_ttft",
        "p95_ttft",
        "mean_lat",
        "p95_lat",
        "cache_hit",
        "hbm_hit",
        "triggers",
        "warm_gb",
        "warm_busy_s",
        "warm_count",
    ]
    return {key: statistics.fmean(row[key] for row in rows) if rows else 0.0 for key in keys}


def evaluate_policy_set(
    args: argparse.Namespace,
    cfg: SimpleNamespace,
    windows: list[Window],
    examples: list[Example],
    scores: dict[tuple[int, int], float],
    threshold: float,
    gate_topk: int,
    baselines: dict,
) -> dict:
    by_window_examples: dict[int, list[Example]] = defaultdict(list)
    for ex in examples:
        by_window_examples[ex.window_id].append(ex)

    rows = {
        "baseline": [],
        "candidate_all": [],
        "oracle_utility": [],
        "oracle_greedy": [],
        "trained_gate": [],
    }
    for window in windows:
        window_id = window.obs[0].window_id
        baseline = baselines[window_id]
        rows["baseline"].append(baseline)
        candidate_ids = set(window.candidates)
        oracle_ids = {ex.request_id for ex in by_window_examples[window_id] if ex.label}
        greedy_ids: set[int] = set()
        greedy_obj = objective(baseline, args)
        for ex in sorted(by_window_examples[window_id], key=lambda item: item.utility, reverse=True):
            if ex.utility <= args.min_utility:
                continue
            trial_ids = greedy_ids | {ex.request_id}
            trial = run_policy(
                cfg,
                window,
                key_blocks=args.key_blocks,
                warm_blocks_count=args.warm_blocks,
                replicas=args.replicas,
                active_ttl_s=args.active_ttl_s,
                trigger_ids=trial_ids,
            )
            trial_obj = objective(trial, args)
            if trial_obj < greedy_obj:
                greedy_obj = trial_obj
                greedy_ids = trial_ids
        trained_ids = select_score_ids(by_window_examples[window_id], scores, threshold, gate_topk)
        for name, ids in [
            ("candidate_all", candidate_ids),
            ("oracle_utility", oracle_ids),
            ("oracle_greedy", greedy_ids),
            ("trained_gate", trained_ids),
        ]:
            rows[name].append(
                run_policy(
                    cfg,
                    window,
                    key_blocks=args.key_blocks,
                    warm_blocks_count=args.warm_blocks,
                    replicas=args.replicas,
                    active_ttl_s=args.active_ttl_s,
                    trigger_ids=ids,
                )
            )
    return {name: aggregate(policy_rows) for name, policy_rows in rows.items()}


def print_eval(title: str, summary: dict, baseline_name: str = "baseline") -> None:
    base = summary[baseline_name]
    print(f"\n{title}")
    print(
        f"{'policy':<18} {'meanTTFT':>9} {'dMean':>9} {'p95TTFT':>9} {'dP95':>9} "
        f"{'meanLat':>9} {'trig':>7} {'warmGB':>8} {'warmBusy':>9}"
    )
    for name, row in summary.items():
        print(
            f"{name:<18} "
            f"{row['mean_ttft']:>8.4f}s "
            f"{row['mean_ttft'] - base['mean_ttft']:>+8.4f}s "
            f"{row['p95_ttft']:>8.4f}s "
            f"{row['p95_ttft'] - base['p95_ttft']:>+8.4f}s "
            f"{row['mean_lat']:>8.4f}s "
            f"{row['triggers']:>7.1f} "
            f"{row['warm_gb']:>7.3f} "
            f"{row['warm_busy_s']:>8.3f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["mooncake_parquet", "burstgpt_csv", "jsonl"], default="mooncake_parquet")
    parser.add_argument("--dataset", default="valeriol29/mooncake-traces")
    parser.add_argument("--config-name", default="mooncake")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--burstgpt-url",
        default="https://github.com/HPMLL/BurstGPT/releases/download/v2.0/BurstGPT_3.csv",
    )
    parser.add_argument("--burstgpt-max-rows", type=int, default=50000)
    parser.add_argument("--burstgpt-starts", type=int, nargs="*")
    parser.add_argument("--burstgpt-sequential", action="store_true")
    parser.add_argument("--jsonl-path")
    parser.add_argument("--jsonl-max-rows", type=int, default=0)
    parser.add_argument("--jsonl-starts", type=int, nargs="*")
    parser.add_argument("--jsonl-sequential", action="store_true")
    parser.add_argument("--jsonl-request-id-field", default="request_id")
    parser.add_argument("--jsonl-timestamp-field", default="timestamp")
    parser.add_argument("--jsonl-delay-field", default="delay")
    parser.add_argument("--jsonl-delay-divisor", type=float, default=1000.0)
    parser.add_argument("--jsonl-input-field", default="input_length")
    parser.add_argument("--jsonl-output-field", default="output_length")
    parser.add_argument("--jsonl-hash-field", default="hash_ids")
    parser.add_argument("--jsonl-session-field", default="session_id")
    parser.add_argument("--jsonl-group-field", default="group_id")
    parser.add_argument("--jsonl-message-field", default="messages")
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--train-windows", type=int, default=3)
    parser.add_argument(
        "--threshold-windows",
        type=int,
        default=0,
        help="Reserve this many pre-test windows for replay threshold/top-k selection instead of selecting on fit windows.",
    )
    parser.add_argument("--rows-per-window", type=int, default=500)
    parser.add_argument("--max-parquet-files", type=int, default=1)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--arrival-scale", type=float, default=0.25)
    parser.add_argument("--key-blocks", type=int, default=8)
    parser.add_argument("--block-tokens", type=int, default=512)
    parser.add_argument("--horizon-s", type=float, default=30.0)
    parser.add_argument("--future-k", type=int, default=3)
    parser.add_argument("--warm-blocks", type=int, default=8)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--active-ttl-s", type=float, default=30.0)
    parser.add_argument("--rdma-gbps", type=float, default=0.0)
    parser.add_argument("--hbm-only", action="store_true", default=True)
    parser.add_argument("--max-batch", type=int, default=256)
    parser.add_argument("--model-preset", choices=["default", "glm52-int4"], default="glm52-int4")
    parser.add_argument("--num-replicas", type=int, default=8)
    parser.add_argument("--gpus-per-replica", type=int, default=8)
    parser.add_argument("--gpu", choices=["H100", "H200", "B200", "B300", "A100"], default="H100")
    parser.add_argument("--imbalance-abs", type=int, default=8)
    parser.add_argument("--imbalance-rel", type=float, default=1.5)
    parser.add_argument("--objective-metric", choices=["mean_ttft", "p95_ttft", "mean_lat", "p95_lat"], default="mean_ttft")
    parser.add_argument("--warm-gb-cost", type=float, default=0.0)
    parser.add_argument("--warm-busy-cost", type=float, default=0.0)
    parser.add_argument("--trigger-cost", type=float, default=0.0)
    parser.add_argument("--min-utility", type=float, default=0.0)
    parser.add_argument("--max-candidates-per-window", type=int, default=80)
    parser.add_argument("--include-cold-candidates", action="store_true")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--threshold-selection", choices=["class_f1", "replay"], default="replay")
    parser.add_argument(
        "--gate-topk-options",
        type=int,
        nargs="*",
        default=[0],
        help="Per-window trained-gate trigger caps to consider during replay threshold selection; 0 means no cap.",
    )
    parser.add_argument("--threshold-grid-step", type=float, default=0.01)
    parser.add_argument(
        "--threshold-score-candidates",
        dest="threshold_score_candidates",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-threshold-score-candidates", dest="threshold_score_candidates", action="store_false")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--out", default="experiments/btb_utility_gate_results.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_windows <= 0 or args.train_windows >= args.windows:
        raise SystemExit("--train-windows must be between 1 and windows-1")
    if args.threshold_windows < 0 or args.threshold_windows >= args.train_windows:
        raise SystemExit("--threshold-windows must be >= 0 and smaller than --train-windows")
    if args.source == "jsonl" and not args.jsonl_path:
        raise SystemExit("--jsonl-path is required when --source jsonl")
    cfg = make_cfg(args)
    cfg.RDMA_ON_ADMIT = False

    print(
        "BTB utility-gate experiment: "
        f"{args.windows} windows x {args.rows_per_window} rows, train_windows={args.train_windows}, "
        f"source={args.source}, dataset={args.dataset}, config={args.config_name}, "
        f"objective={args.objective_metric}"
        f"+{args.warm_gb_cost:g}*warm_gb"
        f"+{args.warm_busy_cost:g}*warm_busy_s"
        f"+{args.trigger_cost:g}*triggers",
        flush=True,
    )
    windows = load_windows(args, cfg)
    examples, baselines = label_examples(args, cfg, windows)
    train, test = split_examples(examples, args.train_windows)
    if not train or not test:
        raise SystemExit("not enough train/test examples")
    threshold_start = args.train_windows - args.threshold_windows if args.threshold_windows else 0
    fit_examples = window_slice_examples(examples, 0, threshold_start) if args.threshold_windows else train
    threshold_examples = (
        window_slice_examples(examples, threshold_start, args.train_windows)
        if args.threshold_windows
        else train
    )
    if not fit_examples:
        raise SystemExit("not enough fit examples")
    if not threshold_examples:
        raise SystemExit("not enough threshold-selection examples")

    print(
        f"\nexamples: train={len(train)} pos={sum(ex.label for ex in train)} "
        f"({sum(ex.label for ex in train) / len(train):.1%}), "
        f"test={len(test)} pos={sum(ex.label for ex in test)} "
        f"({sum(ex.label for ex in test) / len(test):.1%})",
        flush=True,
    )
    if args.threshold_windows:
        print(
            f"fit examples: n={len(fit_examples)} pos={sum(ex.label for ex in fit_examples)} "
            f"({sum(ex.label for ex in fit_examples) / len(fit_examples):.1%}); "
            f"threshold examples: n={len(threshold_examples)} pos={sum(ex.label for ex in threshold_examples)} "
            f"({sum(ex.label for ex in threshold_examples) / len(threshold_examples):.1%})",
            flush=True,
        )

    means, stds = standardizer(fit_examples)
    weights = train_logreg(fit_examples, means, stds, epochs=args.epochs, lr=args.lr, l2=args.l2)
    all_scores = {
        (ex.window_id, ex.request_id): score_example(ex, means, stds, weights)
        for ex in examples
    }
    class_threshold, _ = choose_threshold(threshold_examples, all_scores)
    threshold = class_threshold
    gate_topk = 0
    replay_threshold_info = None
    if args.threshold_selection == "replay":
        threshold_windows = (
            windows[threshold_start : args.train_windows]
            if args.threshold_windows
            else windows[: args.train_windows]
        )
        threshold, gate_topk, _, replay_threshold_info = choose_threshold_by_replay(
            args,
            cfg,
            threshold_windows,
            threshold_examples,
            all_scores,
            baselines,
        )
    test_class_metrics = binary_metrics(test, all_scores, threshold, gate_topk)
    train_class_metrics = binary_metrics(train, all_scores, threshold, gate_topk)
    fit_class_metrics = binary_metrics(fit_examples, all_scores, threshold, gate_topk)
    threshold_class_metrics = binary_metrics(threshold_examples, all_scores, threshold, gate_topk)
    print(f"chosen threshold={threshold:.3f}, gate_topk={gate_topk or 'all'}")
    if args.threshold_selection == "replay":
        base = replay_threshold_info["baseline"]
        best = replay_threshold_info["best"]
        print(
            "threshold selected by train replay: "
            f"objective {objective(base, args):.6f} -> {objective(best, args):.6f}, "
            f"{args.objective_metric} {base[args.objective_metric]:.6f} -> "
            f"{best[args.objective_metric]:.6f}, triggers={best['triggers']:.1f}, "
            f"gate_topk={gate_topk or 'all'}",
            flush=True,
        )
    print(f"train gate metrics: {train_class_metrics}")
    if args.threshold_windows:
        print(f"fit   gate metrics: {fit_class_metrics}")
        print(f"valid gate metrics: {threshold_class_metrics}")
    print(f"test  gate metrics: {test_class_metrics}")

    train_windows = windows[: args.train_windows]
    test_windows = windows[args.train_windows :]
    train_summary = evaluate_policy_set(args, cfg, train_windows, train, all_scores, threshold, gate_topk, baselines)
    test_summary = evaluate_policy_set(args, cfg, test_windows, test, all_scores, threshold, gate_topk, baselines)
    print_eval("Train-window replay", train_summary)
    print_eval("Held-out replay", test_summary)

    coeffs = [{"feature": "bias", "weight": weights[0]}] + [
        {"feature": name, "weight": weight, "mean": mean, "std": std}
        for name, weight, mean, std in zip(FEATURE_NAMES, weights[1:], means, stds)
    ]
    coeffs = sorted(coeffs, key=lambda row: abs(row["weight"]), reverse=True)

    payload = {
        "args": vars(args),
        "windows": [window.ident for window in windows],
        "class_metrics": {
            "train": train_class_metrics,
            "fit": fit_class_metrics,
            "threshold": threshold_class_metrics,
            "test": test_class_metrics,
        },
        "threshold": threshold,
        "gate_topk": gate_topk,
        "class_f1_threshold": class_threshold,
        "threshold_selection": args.threshold_selection,
        "threshold_start_window": threshold_start,
        "objective": {
            "metric": args.objective_metric,
            "warm_gb_cost": args.warm_gb_cost,
            "warm_busy_cost": args.warm_busy_cost,
            "trigger_cost": args.trigger_cost,
        },
        "replay_threshold_info": replay_threshold_info,
        "train_summary": train_summary,
        "test_summary": test_summary,
        "examples": [
            {
                "window_id": ex.window_id,
                "request_id": ex.request_id,
                "utility": ex.utility,
                "label": ex.label,
                "score": all_scores[(ex.window_id, ex.request_id)],
            }
            for ex in examples
        ],
        "model": {
            "type": "standardized_logistic_regression",
            "feature_names": FEATURE_NAMES,
            "means": means,
            "stds": stds,
            "weights": weights,
            "top_coefficients": coeffs[:12],
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
