#!/usr/bin/env python3
"""End-to-end ART evaluation for predictor-triggered KV warming.

This script answers the practical question:

  If the burst predictor fires online, do KV warming actions actually improve
  TTFT/latency across real ART windows after false positives and warm costs?

It reuses the saved pure-Python logistic regression from
art_burst_model.json, samples real ART parquet row groups, and compares:

  - cache_aware_no_remote: cache-affinity router, no peer KV transfer
  - least_load_no_remote: load-only router, recomputes cold prefixes
  - reactive_copy_rdma: load-only router, peer KV can be fetched on admission
  - predict_copy_bN_tT: predictor triggers background RDMA copies
  - predict_fake_bN_tT: predictor triggers fake-prefill warming
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from workload import Request


def load_local_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BTB_DIR = Path(__file__).resolve().parent
predictor = load_local_module("btb_predict_bursts_art", BTB_DIR / "predict_bursts_art.py")
synthetic = load_local_module("btb_synthetic_warming", BTB_DIR / "run.py")


@dataclass(frozen=True)
class PolicySpec:
    name: str
    route: str
    allow_remote_on_admit: bool = False
    action: str | None = None
    threshold: float = 1.0
    replicas: int = 4
    warm_blocks: int = 8


@dataclass
class PendingWarm:
    ready: float
    node: object
    key: tuple[str, ...]
    blocks: list[str]
    kind: str
    bytes: int
    duration: float


@dataclass
class Window:
    ident: str
    obs: list
    requests: list[Request]
    scores: dict[int, float]
    labels: dict[int, tuple[int, int]]


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[idx]


def make_cfg(
    rdma_gbps_per_gpu: float,
    arrival_scale: float,
    hbm_only: bool,
    block_tokens: int,
) -> SimpleNamespace:
    cfg = synthetic.make_cfg(rdma_gbps_per_gpu)
    cfg.ARRIVAL_SCALE = arrival_scale
    cfg.HBM_ONLY = hbm_only
    cfg.BLOCK_TOKENS = block_tokens
    return cfg


def scaled_obs(obs: list, scale: float) -> list:
    if scale == 1.0:
        return obs
    return [
        predictor.Obs(
            window_id=item.window_id,
            row_in_window=item.row_in_window,
            request_id=item.request_id,
            t=item.t * scale,
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
            blocks=item.blocks,
            key=item.key,
            first_block=item.first_block,
            system_hash=item.system_hash,
            message_bytes=item.message_bytes,
        )
        for item in obs
    ]


def build_feature_rows(obs: list) -> list[dict[str, float]]:
    total_history = {w: deque() for w in predictor.WINDOWS_S}
    key_history = {w: defaultdict(deque) for w in predictor.WINDOWS_S}
    first_history = {w: defaultdict(deque) for w in predictor.WINDOWS_S}
    system_history = {w: defaultdict(deque) for w in predictor.WINDOWS_S}
    seen_key = defaultdict(int)
    seen_first = defaultdict(int)
    last_key = {}
    last_first = {}
    prev_t = None
    rows = []

    for item in obs:
        interarrival = 0.0 if prev_t is None else max(0.0, item.t - prev_t)
        feats = {
            "input_tokens_log": predictor.log1p_cap(item.input_tokens),
            "prefix_blocks_log": predictor.log1p_cap(len(item.blocks)),
            "message_bytes_log": predictor.log1p_cap(item.message_bytes),
            "interarrival_s_log": predictor.log1p_cap(interarrival, 60.0),
            "same_key_seen": float(seen_key[item.key]) if item.key else 0.0,
            "same_first_seen": float(seen_first[item.first_block]) if item.first_block else 0.0,
        }
        feats["same_key_seen_log"] = predictor.log1p_cap(feats["same_key_seen"])
        feats["same_first_seen_log"] = predictor.log1p_cap(feats["same_first_seen"])

        if item.key and item.key in last_key:
            feats["time_since_same_key_log"] = predictor.log1p_cap(item.t - last_key[item.key], 3600.0)
            feats["no_prior_same_key"] = 0.0
        else:
            feats["time_since_same_key_log"] = predictor.log1p_cap(3600.0)
            feats["no_prior_same_key"] = 1.0

        if item.first_block and item.first_block in last_first:
            feats["time_since_same_first_log"] = predictor.log1p_cap(
                item.t - last_first[item.first_block], 3600.0
            )
            feats["no_prior_same_first"] = 0.0
        else:
            feats["time_since_same_first_log"] = predictor.log1p_cap(3600.0)
            feats["no_prior_same_first"] = 1.0

        for w in predictor.WINDOWS_S:
            total_q = total_history[w]
            predictor.pop_old(total_q, item.t, w)
            total = len(total_q)
            same_key = predictor.recent_count(key_history[w], item.key, item.t, w)
            same_first = predictor.recent_count(first_history[w], item.first_block, item.t, w)
            same_system = predictor.recent_count(system_history[w], item.system_hash, item.t, w)
            suffix = f"{int(w)}s"
            feats[f"total_{suffix}"] = float(total)
            feats[f"total_{suffix}_log"] = predictor.log1p_cap(total)
            feats[f"same_key_{suffix}"] = float(same_key)
            feats[f"same_key_{suffix}_log"] = predictor.log1p_cap(same_key)
            feats[f"same_first_{suffix}"] = float(same_first)
            feats[f"same_first_{suffix}_log"] = predictor.log1p_cap(same_first)
            feats[f"same_system_{suffix}"] = float(same_system)
            feats[f"same_system_{suffix}_log"] = predictor.log1p_cap(same_system)

        rows.append(feats)

        for w in predictor.WINDOWS_S:
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

    return rows


def load_model(path: Path) -> dict:
    return json.loads(path.read_text())


def score_feature_row(model: dict, features: dict[str, float]) -> float:
    spec = model["model"]
    z = spec["weights"][0]
    for name, mean, std, weight in zip(
        spec["feature_names"],
        spec["means"],
        spec["stds"],
        spec["weights"][1:],
    ):
        z += weight * ((features.get(name, 0.0) - mean) / std)
    return sigmoid(z)


def requests_from_obs(obs: list, cfg: SimpleNamespace) -> list[Request]:
    requests = []
    for i, item in enumerate(obs):
        blocks = list(item.blocks)
        input_tokens = max(1, int(item.input_tokens))
        output_tokens = max(1, int(item.output_tokens))
        extra = math.ceil((input_tokens + output_tokens) / cfg.BLOCK_TOKENS) - len(blocks)
        cache_blocks = blocks + [f"{item.request_id}#o{j}" for j in range(max(0, extra))]
        group = ":".join(item.key[:2]) if item.key else item.first_block[:12]
        requests.append(
            Request(
                id=i,
                arrival=item.t,
                group=group,
                prefix_tokens=min(len(blocks) * cfg.BLOCK_TOKENS, input_tokens),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                blocks=blocks,
                cache_blocks=cache_blocks,
            )
        )
    return requests


def load_windows(args: argparse.Namespace, cfg: SimpleNamespace, model: dict) -> list[Window]:
    import fsspec
    import pyarrow.parquet as pq

    rng = random.Random(args.seed)
    files = predictor.parquet_files(args.dataset, args.split, args.config_name)[: args.max_parquet_files]
    fs = fsspec.filesystem("https")
    candidates = []
    metadata = {}
    used_training_offsets = set(model.get("data", {}).get("offsets", []))

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
            f"no parquet row groups can supply {args.rows_per_window} rows"
        )

    rng.shuffle(candidates)
    windows = []
    if len(candidates) >= args.windows:
        chosen_candidates = candidates
    else:
        chosen_candidates = [rng.choice(candidates) for _ in range(args.windows)]
    for file_idx, rg, nrows in chosen_candidates:
        if len(windows) >= args.windows:
            break
        entry = metadata[file_idx]["entry"]
        start = rng.randrange(0, nrows - args.rows_per_window + 1)
        ident = f"{entry['filename']}:rg{rg}:start{start}"
        if ident in used_training_offsets:
            continue
        with fs.open(entry["url"], "rb", block_size=1 << 20) as fh:
            pf = pq.ParquetFile(fh)
            available = set(pf.schema_arrow.names)
            columns = [name for name in predictor.COLUMNS if name in available]
            table = pf.read_row_group(rg, columns=columns)
        rows = table.slice(start, args.rows_per_window).to_pylist()
        obs = predictor.normalize_rows(rows, len(windows), args.key_blocks)
        obs = scaled_obs(obs, args.arrival_scale)
        features = build_feature_rows(obs)
        labels = predictor.label_window(obs, args.horizon_s, args.future_k)
        if args.score_mode == "oracle":
            scores = {i: float(labels.get(i, (0, 0))[0]) for i in range(len(features))}
        elif args.score_mode == "oracle_count":
            scores = {
                i: min(1.0, labels.get(i, (0, 0))[1] / max(1, args.future_k))
                for i in range(len(features))
            }
        else:
            scores = {i: score_feature_row(model, feats) for i, feats in enumerate(features)}
        requests = requests_from_obs(obs, cfg)
        positives = sum(label for label, _ in labels.values())
        print(
            f"window {len(windows):02d} {ident:<24} rows={len(rows):<5} "
            f"labelable={len(labels):<5} pos={positives:<5} span={obs[-1].t if obs else 0:.1f}s",
            flush=True,
        )
        windows.append(Window(ident, obs, requests, scores, labels))

    if len(windows) < args.windows:
        raise RuntimeError(f"only loaded {len(windows)} windows; requested {args.windows}")
    return windows


def prefix_key(req: Request, key_blocks: int) -> tuple[str, ...]:
    if len(req.blocks) < key_blocks:
        return ()
    return tuple(req.blocks[:key_blocks])


def warm_blocks(req: Request, blocks: int) -> list[str]:
    return list(req.blocks[: min(blocks, len(req.blocks))])


def full_nodes(nodes: list, blocks: list[str]) -> list:
    if not blocks:
        return []
    want = len(blocks)
    return [node for node in nodes if synthetic.local_blocks(node, blocks) >= want]


def node_load(node, now: float) -> tuple[float, int, str]:
    return synthetic.load_key(node, now)


def choose_cache_aware(req: Request, nodes: list, now: float, cfg: SimpleNamespace):
    loads = [len(node.running) + len(node.waiting) for node in nodes]
    if max(loads) > cfg.IMBALANCE_ABS and max(loads) > cfg.IMBALANCE_REL * min(loads):
        return min(nodes, key=lambda node: node_load(node, now))
    return min(nodes, key=lambda node: (-synthetic.local_blocks(node, req.blocks), node_load(node, now)))


def choose_node(
    policy: PolicySpec,
    req: Request,
    nodes: list,
    now: float,
    cfg: SimpleNamespace,
    active_keys: dict[tuple[str, ...], float],
    key_blocks: int,
) -> object:
    if policy.route == "least_load":
        return min(nodes, key=lambda node: node_load(node, now))

    if policy.route == "cache_aware":
        return choose_cache_aware(req, nodes, now, cfg)

    if policy.route == "predictive":
        key = prefix_key(req, key_blocks)
        if key and active_keys.get(key, -1.0) >= now:
            blocks = warm_blocks(req, policy.warm_blocks)
            candidates = full_nodes(nodes, blocks)
            if candidates:
                return min(candidates, key=lambda node: node_load(node, now))
        return choose_cache_aware(req, nodes, now, cfg)

    raise ValueError(f"unknown route: {policy.route}")


def choose_seed_target(
    policy: PolicySpec,
    req: Request,
    nodes: list,
    now: float,
    key_blocks: int,
    planned: set[tuple[tuple[str, ...], str]],
) -> object | None:
    key = prefix_key(req, key_blocks)
    blocks = warm_blocks(req, policy.warm_blocks)
    if not key or not blocks:
        return None
    existing = full_nodes(nodes, blocks)
    planned_count = sum(1 for node in nodes if (key, node.name) in planned)
    if len(existing) + planned_count >= policy.replicas:
        return None
    candidates = [
        node
        for node in nodes
        if node not in existing and (key, node.name) not in planned
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda node: node_load(node, now))


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


def schedule_action(
    policy: PolicySpec,
    req: Request,
    chosen,
    nodes: list,
    now: float,
    key_blocks: int,
    planned: set[tuple[tuple[str, ...], str]],
) -> tuple[list[PendingWarm], dict[str, float]]:
    key = prefix_key(req, key_blocks)
    blocks = warm_blocks(req, policy.warm_blocks)
    if not key or not blocks or not policy.action:
        return [], {"warm_bytes": 0, "warm_busy_s": 0.0, "warm_count": 0}

    warm_bytes = 0
    warm_busy_s = 0.0
    pending = []
    existing = full_nodes(nodes, blocks)
    planned_count = sum(1 for node in nodes if (key, node.name) in planned)

    if policy.action == "copy":
        if not existing:
            return [], {"warm_bytes": 0, "warm_busy_s": 0.0, "warm_count": 0}
        needed = max(0, policy.replicas - len(existing) - planned_count)
        targets = [
            node
            for node in nodes
            if node not in existing and (key, node.name) not in planned
        ]
        targets = sorted(targets, key=lambda node: node_load(node, now))[:needed]
        for target in targets:
            if target.tier_bw["rdma"] <= 0:
                continue
            nbytes = len(blocks) * target.block_bytes
            duration = target.load_time(nbytes, "rdma")
            pending.append(PendingWarm(now + duration, target, key, blocks, "copy", nbytes, duration))
            planned.add((key, target.name))
            warm_bytes += nbytes

    elif policy.action == "fake_prefill":
        implicit_chosen = 0 if chosen in existing else 1
        needed = max(0, policy.replicas - len(existing) - planned_count - implicit_chosen)
        targets = [
            node
            for node in nodes
            if node is not chosen and node not in existing and (key, node.name) not in planned
        ]
        targets = sorted(targets, key=lambda node: node_load(node, now))[:needed]
        tokens = len(blocks) * config.BLOCK_TOKENS
        for target in targets:
            start = max(target.now, now)
            duration = target.prefill_time(tokens)
            target.now = start + duration
            target.busy += duration
            nbytes = len(blocks) * target.block_bytes
            pending.append(PendingWarm(target.now, target, key, blocks, "fake_prefill", nbytes, duration))
            planned.add((key, target.name))
            warm_bytes += nbytes
            warm_busy_s += duration
    else:
        raise ValueError(f"unknown action: {policy.action}")

    return pending, {
        "warm_bytes": warm_bytes,
        "warm_busy_s": warm_busy_s,
        "warm_count": len(pending),
    }


def run_policy(
    policy: PolicySpec,
    cfg: SimpleNamespace,
    window: Window,
    *,
    key_blocks: int,
    active_ttl_s: float,
) -> dict:
    nodes = [synthetic.Node(name, spec, n_gpus, cfg) for name, spec, n_gpus in cfg.CLUSTER]
    events = []
    pending: list[PendingWarm] = []
    planned: set[tuple[tuple[str, ...], str]] = set()
    active_keys: dict[tuple[str, ...], float] = {}
    stats = {
        "triggers": 0,
        "labelable_triggers": 0,
        "trigger_tp": 0,
        "total_positives": sum(label for label, _ in window.labels.values()),
        "warm_bytes": 0,
        "warm_busy_s": 0.0,
        "warm_count": 0,
        "ready_warms": 0,
    }

    for req in window.requests:
        for node in nodes:
            synthetic.admit_and_decode_until(
                node,
                req.arrival,
                nodes,
                cfg,
                events,
                allow_remote=policy.allow_remote_on_admit,
            )
        pending, ready = process_ready_warms(pending, req.arrival, planned)
        stats["ready_warms"] += ready

        now = req.arrival
        active_keys = {key: until for key, until in active_keys.items() if until >= now}
        key = prefix_key(req, key_blocks)
        score = window.scores.get(req.id, 0.0)
        if policy.action and key and score >= policy.threshold:
            active_keys[key] = max(active_keys.get(key, now), now + active_ttl_s)
            stats["triggers"] += 1
            if req.id in window.labels:
                stats["labelable_triggers"] += 1
                stats["trigger_tp"] += window.labels[req.id][0]

        chosen = None
        if policy.action == "seed_real" and key and active_keys.get(key, -1.0) >= now:
            chosen = choose_seed_target(policy, req, nodes, now, key_blocks, planned)
            if chosen is not None:
                planned.add((key, chosen.name))
                stats["warm_count"] += 1
                stats["warm_bytes"] += len(warm_blocks(req, policy.warm_blocks)) * chosen.block_bytes

        if chosen is None:
            chosen = choose_node(policy, req, nodes, now, cfg, active_keys, key_blocks)
        chosen.waiting.append(req)

        if policy.action and policy.action != "seed_real" and key and active_keys.get(key, -1.0) >= now:
            new_pending, warm_stats = schedule_action(
                policy, req, chosen, nodes, now, key_blocks, planned
            )
            pending.extend(new_pending)
            stats["warm_bytes"] += warm_stats["warm_bytes"]
            stats["warm_busy_s"] += warm_stats["warm_busy_s"]
            stats["warm_count"] += warm_stats["warm_count"]

    while pending:
        next_ready = min(warm.ready for warm in pending)
        for node in nodes:
            synthetic.admit_and_decode_until(
                node,
                next_ready,
                nodes,
                cfg,
                events,
                allow_remote=policy.allow_remote_on_admit,
            )
        pending, ready = process_ready_warms(pending, next_ready, planned)
        stats["ready_warms"] += ready

    for node in nodes:
        synthetic.admit_and_decode_until(
            node, None, nodes, cfg, events, allow_remote=policy.allow_remote_on_admit
        )

    events.sort(key=lambda event: event["id"])
    return summarize(policy, cfg, nodes, events, stats)


def summarize(policy: PolicySpec, cfg: SimpleNamespace, nodes: list, events: list[dict], stats: dict) -> dict:
    ttft = [event["start"] - event["arrival"] + event["reuse"] + event["prefill"] for event in events]
    queue = [event["start"] - event["arrival"] for event in events]
    lat = [event["finish"] - event["arrival"] for event in events]
    prefix_tok = sum(event["prefix_tokens"] for event in events)
    hit_tok = sum(event["hit"] for event in events)
    hbm_tok = sum(event["hit"] for event in events if event["tier"] == "hbm")
    out_tok = sum(event["output_tokens"] for event in events)
    span = max((event["finish"] for event in events), default=0.0)
    trigger_precision = (
        stats["trigger_tp"] / stats["labelable_triggers"] if stats["labelable_triggers"] else 0.0
    )
    trigger_recall = (
        stats["trigger_tp"] / stats["total_positives"] if stats["total_positives"] else 0.0
    )
    return {
        "policy": policy.name,
        "requests": len(events),
        "span": span,
        "mean_ttft": statistics.fmean(ttft) if ttft else 0.0,
        "p50_ttft": pct(ttft, 0.50),
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
        "trigger_precision": trigger_precision,
        "trigger_recall": trigger_recall,
        "warm_gb": stats["warm_bytes"] / config.GB,
        "warm_busy_s": stats["warm_busy_s"],
        "warm_count": stats["warm_count"],
        "ready_warms": stats["ready_warms"],
    }


def policy_specs(args: argparse.Namespace, model: dict) -> list[PolicySpec]:
    specs = [
        PolicySpec("cache_aware_no_remote", "cache_aware", allow_remote_on_admit=False),
        PolicySpec("least_load_no_remote", "least_load", allow_remote_on_admit=False),
    ]
    if args.rdma_gbps > 0:
        specs.append(PolicySpec("reactive_copy_rdma", "least_load", allow_remote_on_admit=True))
    thresholds = args.thresholds or [float(model["model"]["threshold"])]
    for warm_blocks in args.warm_blocks:
        for threshold in thresholds:
            suffix = f"b{warm_blocks}_t{threshold:.2f}"
            if args.rdma_gbps > 0:
                specs.append(
                    PolicySpec(
                        f"prewarm_reactive_copy_{suffix}",
                        "least_load",
                        allow_remote_on_admit=True,
                        action="copy",
                        threshold=threshold,
                        replicas=args.replicas,
                        warm_blocks=warm_blocks,
                    )
                )
                specs.append(
                    PolicySpec(
                        f"predict_copy_{suffix}",
                        "predictive",
                        allow_remote_on_admit=False,
                        action="copy",
                        threshold=threshold,
                        replicas=args.replicas,
                        warm_blocks=warm_blocks,
                    )
                )
            if args.include_fake_prefill or args.rdma_gbps <= 0:
                specs.append(
                    PolicySpec(
                        f"predict_fake_{suffix}",
                        "predictive",
                        allow_remote_on_admit=False,
                        action="fake_prefill",
                        threshold=threshold,
                        replicas=args.replicas,
                        warm_blocks=warm_blocks,
                    )
                )
            if args.include_real_seed or args.rdma_gbps <= 0:
                specs.append(
                    PolicySpec(
                        f"seed_real_b{warm_blocks}_t{threshold:.2f}",
                        "predictive",
                        allow_remote_on_admit=False,
                        action="seed_real",
                        threshold=threshold,
                        replicas=args.replicas,
                        warm_blocks=warm_blocks,
                    )
                )
    return specs


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = [
        "mean_ttft",
        "p95_ttft",
        "p99_ttft",
        "mean_queue",
        "p95_queue",
        "mean_lat",
        "p95_lat",
        "p99_lat",
        "cache_hit",
        "hbm_hit",
        "throughput",
        "util",
        "critical_rdma_s",
        "triggers",
        "trigger_precision",
        "trigger_recall",
        "warm_gb",
        "warm_busy_s",
        "warm_count",
    ]
    return {key: statistics.fmean(row[key] for row in rows) for key in keys}


def print_summary(summary: dict[str, dict], deltas: dict[str, dict], baseline: str) -> None:
    print(f"\nAggregate per-window means (delta vs {baseline})")
    print(
        f"{'policy':<28} {'meanTTFT':>9} {'dTTFT':>9} {'p95TTFT':>9} {'dP95':>9} "
        f"{'meanLat':>9} {'dLat':>9} {'hit':>6} {'trigP/R':>11} {'warmGB':>8}"
    )
    for policy, row in summary.items():
        delta = deltas.get(policy, {})
        print(
            f"{policy:<28} "
            f"{row['mean_ttft']:>8.3f}s "
            f"{delta.get('mean_ttft', 0.0):>+8.3f}s "
            f"{row['p95_ttft']:>8.3f}s "
            f"{delta.get('p95_ttft', 0.0):>+8.3f}s "
            f"{row['mean_lat']:>8.3f}s "
            f"{delta.get('mean_lat', 0.0):>+8.3f}s "
            f"{row['cache_hit']:>5.0%} "
            f"{row['trigger_precision']:>4.2f}/{row['trigger_recall']:<4.2f} "
            f"{row['warm_gb']:>7.2f}"
        )


def paired_deltas(results: dict[str, list[dict]], baseline: str) -> dict[str, dict]:
    out = {}
    base = results[baseline]
    for policy, rows in results.items():
        if policy == baseline:
            continue
        deltas = {}
        for metric in ["mean_ttft", "p95_ttft", "mean_lat", "p95_lat"]:
            vals = [row[metric] - b[metric] for row, b in zip(rows, base)]
            deltas[metric] = statistics.fmean(vals)
            deltas[f"{metric}_wins"] = sum(1 for val in vals if val < 0)
        out[policy] = deltas
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="bite-the-bullet/art_burst_model.json")
    parser.add_argument("--dataset", default=config.DATASET)
    parser.add_argument("--config-name")
    parser.add_argument("--split", default=config.DATASET_SPLIT)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--rows-per-window", type=int, default=1000)
    parser.add_argument("--max-parquet-files", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--arrival-scale", type=float, default=1.0)
    parser.add_argument("--rdma-gbps", type=float, default=50.0)
    parser.add_argument("--hbm-only", action="store_true")
    parser.add_argument("--key-blocks", type=int, default=8)
    parser.add_argument("--block-tokens", type=int, default=config.BLOCK_TOKENS)
    parser.add_argument("--horizon-s", type=float, default=30.0)
    parser.add_argument("--future-k", type=int, default=3)
    parser.add_argument("--thresholds", type=float, nargs="*")
    parser.add_argument("--score-mode", choices=["model", "oracle", "oracle_count"], default="model")
    parser.add_argument("--warm-blocks", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--active-ttl-s", type=float, default=30.0)
    parser.add_argument("--include-fake-prefill", action="store_true")
    parser.add_argument("--include-real-seed", action="store_true")
    parser.add_argument("--baseline", default="reactive_copy_rdma")
    parser.add_argument("--out", default="bite-the-bullet/art_warming_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(Path(args.model))
    cfg = make_cfg(args.rdma_gbps, args.arrival_scale, args.hbm_only, args.block_tokens)
    baseline = args.baseline if args.rdma_gbps > 0 else "least_load_no_remote"
    specs = policy_specs(args, model)
    if baseline not in {spec.name for spec in specs}:
        baseline = specs[0].name

    print(
        "ART warming eval: "
        f"{args.windows} windows x {args.rows_per_window} rows, "
        f"arrival_scale={args.arrival_scale:g}, rdma={args.rdma_gbps:g} GB/s/GPU, "
        f"hbm_only={args.hbm_only}, score_mode={args.score_mode}, "
        f"warm_blocks={args.warm_blocks}, baseline={baseline}",
        flush=True,
    )
    windows = load_windows(args, cfg, model)

    results: dict[str, list[dict]] = {spec.name: [] for spec in specs}
    per_window = []
    for i, window in enumerate(windows):
        print(f"simulate window {i:02d} {window.ident}", flush=True)
        window_rows = {"ident": window.ident, "policies": {}}
        for spec in specs:
            row = run_policy(
                spec,
                cfg,
                window,
                key_blocks=args.key_blocks,
                active_ttl_s=args.active_ttl_s,
            )
            results[spec.name].append(row)
            window_rows["policies"][spec.name] = row
        per_window.append(window_rows)

    summary = {policy: aggregate(rows) for policy, rows in results.items()}
    deltas = paired_deltas(results, baseline)
    print_summary(summary, deltas, baseline)

    payload = {
        "args": vars(args),
        "baseline": baseline,
        "summary": summary,
        "paired_deltas": deltas,
        "per_window": per_window,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
