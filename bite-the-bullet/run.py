#!/usr/bin/env python3
"""Predictive KV warming experiment.

This is a deliberately small experiment for the "bite the bullet" idea:

1. A first request warms a long shared prefix on node0.
2. A burst of many requests with that same prefix arrives a few seconds later.
3. We compare cache-only routing, load-only recomputation, reactive RDMA copy,
   predictive RDMA copy, and predictive fake-prefill warming.

The main simulator allows remote KV loads during admission. For this question
we need that knob exposed, because "copy after the request is already waiting"
and "warm a replica before the burst" are different policies.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from gpu import Node
from workload import Request


def make_cfg(rdma_gbps_per_gpu: float) -> SimpleNamespace:
    """Clone repo defaults and override peer bandwidth for this run."""
    cfg = SimpleNamespace(**config.as_dict())
    cfg.DISK_CACHE = False
    cfg.HBM_ONLY = False
    cluster = []
    for name, spec, n_gpus in cfg.CLUSTER:
        cluster.append((name, replace(spec, rdma_bw=rdma_gbps_per_gpu * config.GB), n_gpus))
    cfg.CLUSTER = cluster
    return cfg


def ceil_blocks(tokens: int, block_tokens: int) -> int:
    return math.ceil(tokens / block_tokens)


def build_hot_burst(
    cfg: SimpleNamespace,
    *,
    shared_prefix_tokens: int,
    burst: int,
    burst_start_s: float,
    burst_window_s: float,
    suffix_tokens: int,
    output_tokens: int,
) -> list[Request]:
    """Synthetic trace: one seed request, then a hot-prefix burst."""
    hot_block_count = ceil_blocks(shared_prefix_tokens, cfg.BLOCK_TOKENS)
    shared_prefix_tokens = hot_block_count * cfg.BLOCK_TOKENS
    hot_blocks = [f"hot:{i}" for i in range(hot_block_count)]

    def req(req_id: int, arrival: float, suffix_id: str, out: int) -> Request:
        suffix_blocks = ceil_blocks(suffix_tokens, cfg.BLOCK_TOKENS)
        blocks = hot_blocks + [f"{suffix_id}:suffix:{j}" for j in range(suffix_blocks)]
        input_tokens = shared_prefix_tokens + suffix_blocks * cfg.BLOCK_TOKENS
        extra = ceil_blocks(input_tokens + out, cfg.BLOCK_TOKENS) - len(blocks)
        cache_blocks = blocks + [f"{suffix_id}:out:{j}" for j in range(max(0, extra))]
        return Request(
            id=req_id,
            arrival=arrival,
            group="hot",
            prefix_tokens=input_tokens,
            input_tokens=input_tokens,
            output_tokens=out,
            blocks=blocks,
            cache_blocks=cache_blocks,
        )

    requests = [req(0, 0.0, "seed", min(output_tokens, 32))]
    if burst == 1:
        arrivals = [burst_start_s]
    else:
        arrivals = [
            burst_start_s + burst_window_s * i / (burst - 1)
            for i in range(burst)
        ]
    for i, arrival in enumerate(arrivals, start=1):
        requests.append(req(i, arrival, f"burst:{i}", output_tokens))
    return requests


class Seq:
    __slots__ = ("req", "start", "reuse", "prefill", "hit", "tier", "left", "context")

    def __init__(self, req: Request, start: float, reuse: float, prefill: float, hit: int, tier: str):
        self.req = req
        self.start = start
        self.reuse = reuse
        self.prefill = prefill
        self.hit = hit
        self.tier = tier
        self.left = req.output_tokens
        self.context = req.input_tokens


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


def prefix_source(
    req: Request,
    node: Node,
    nodes: list[Node],
    cfg: SimpleNamespace,
    *,
    allow_remote: bool,
) -> tuple[int, float, str]:
    """Best cached leading run visible to this node."""
    hbm_n, ram_n = node.match(req.blocks)
    if getattr(cfg, "HBM_ONLY", False):
        candidates = [(hbm_n, 0.0, "hbm")]
    else:
        candidates = [
            (
                hbm_n + ram_n,
                node.load_time(ram_n * node.block_bytes, "ram"),
                "hbm" if ram_n == 0 else "ram",
            )
        ]

    if allow_remote and node.tier_bw["rdma"] > 0:
        remote_n = max((sum(nd.match(req.blocks)) for nd in nodes if nd is not node), default=0)
        candidates.append(
            (remote_n, node.load_time(remote_n * node.block_bytes, "rdma"), "rdma")
        )

    return max(candidates, key=lambda c: (c[0], -c[1]))


def admit_and_decode_until(
    node: Node,
    until: float | None,
    nodes: list[Node],
    cfg: SimpleNamespace,
    events: list[dict],
    *,
    allow_remote: bool,
) -> None:
    """One-node continuous batching loop with an explicit remote-KV switch."""
    while True:
        while node.waiting and len(node.running) < cfg.MAX_BATCH:
            req = node.waiting[0]
            need = req.input_tokens + req.output_tokens
            used = sum(s.req.input_tokens + s.req.output_tokens for s in node.running)
            if used + need > node.kv_budget:
                if not node.running:
                    raise ValueError(
                        f"request {req.id} needs {need} KV tokens but {node.name} "
                        f"has only {node.kv_budget} tokens of post-weight KV headroom"
                    )
                break

            node.waiting.popleft()
            n_blocks, load_s, tier = prefix_source(
                req, node, nodes, cfg, allow_remote=allow_remote
            )
            hit = min(n_blocks * cfg.BLOCK_TOKENS, req.prefix_tokens)

            # Only count it as a hit if loading the cached prefix beats recompute.
            # HBM load is zero, so local cache always wins.
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

            node.running.append(Seq(req, node.now, reuse, prefill, used_hit, used_tier))
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
        kv_tokens = sum(s.context for s in node.running)
        steps = min(s.left for s in node.running)
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


def local_blocks(node: Node, blocks: list[str]) -> int:
    return sum(node.match(blocks))


def full_prefix_nodes(nodes: list[Node], hot_blocks: list[str]) -> list[Node]:
    want = len(hot_blocks)
    return [node for node in nodes if local_blocks(node, hot_blocks) >= want]


def load_key(node: Node, now: float) -> tuple[float, int, str]:
    busy_backlog = max(0.0, node.now - now)
    return (len(node.running) + len(node.waiting) + busy_backlog, len(node.waiting), node.name)


@dataclass
class PendingWarm:
    ready: float
    node: Node
    blocks: list[str]
    kind: str
    duration: float
    bytes: int


@dataclass(frozen=True)
class Policy:
    name: str
    route: str
    allow_remote_on_admit: bool = False
    warm_kind: str | None = None
    replicas: int = 1
    lead_s: float = 0.5


def schedule_warm(
    policy: Policy,
    now: float,
    nodes: list[Node],
    hot_blocks: list[str],
    shared_prefix_tokens: int,
) -> tuple[list[PendingWarm], dict]:
    """Schedule proactive replicas. Replicas includes the original source."""
    if not policy.warm_kind or policy.replicas <= 1:
        return [], {"warm_targets": 0, "warm_bytes": 0, "warm_busy_s": 0.0}

    sources = full_prefix_nodes(nodes, hot_blocks)
    source = sources[0] if sources else nodes[0]
    targets = [node for node in nodes if node is not source]
    targets = sorted(targets, key=lambda node: load_key(node, now))[: max(0, policy.replicas - 1)]

    pending: list[PendingWarm] = []
    warm_bytes = 0
    warm_busy_s = 0.0
    for target in targets:
        if policy.warm_kind == "copy":
            if target.tier_bw["rdma"] <= 0:
                continue
            nbytes = len(hot_blocks) * target.block_bytes
            duration = target.load_time(nbytes, "rdma")
            pending.append(PendingWarm(now + duration, target, hot_blocks, "copy", duration, nbytes))
            warm_bytes += nbytes
        elif policy.warm_kind == "fake_prefill":
            # A fake request cannot use peer KV in the no-RDMA case. It burns
            # prefill compute early and inserts the hot prefix when complete.
            start = max(target.now, now)
            duration = target.prefill_time(shared_prefix_tokens)
            target.now = start + duration
            target.busy += duration
            nbytes = len(hot_blocks) * target.block_bytes
            pending.append(PendingWarm(target.now, target, hot_blocks, "fake_prefill", duration, nbytes))
            warm_busy_s += duration
            warm_bytes += nbytes
        else:
            raise ValueError(f"unknown warm kind: {policy.warm_kind}")

    return pending, {
        "warm_targets": len(pending),
        "warm_bytes": warm_bytes,
        "warm_busy_s": warm_busy_s,
    }


def process_ready_warms(pending: list[PendingWarm], now: float) -> tuple[list[PendingWarm], int]:
    ready_count = 0
    still_pending = []
    for warm in pending:
        if warm.ready <= now + 1e-12:
            warm.node.insert(warm.blocks)
            ready_count += 1
        else:
            still_pending.append(warm)
    return still_pending, ready_count


def choose_node(policy: Policy, req: Request, nodes: list[Node], hot_blocks: list[str], now: float) -> Node:
    if req.id == 0:
        return nodes[0]

    if policy.route == "cache_only":
        return min(nodes, key=lambda node: (-local_blocks(node, req.blocks), load_key(node, now)))

    if policy.route == "least_load":
        return min(nodes, key=lambda node: load_key(node, now))

    if policy.route == "warmed_cache":
        candidates = full_prefix_nodes(nodes, hot_blocks)
        if candidates:
            return min(candidates, key=lambda node: load_key(node, now))
        return min(nodes, key=lambda node: (-local_blocks(node, req.blocks), load_key(node, now)))

    raise ValueError(f"unknown route mode: {policy.route}")


def run_policy(
    policy: Policy,
    cfg: SimpleNamespace,
    requests: list[Request],
    *,
    shared_prefix_tokens: int,
    burst_start_s: float,
) -> dict:
    nodes = [Node(name, spec, n_gpus, cfg) for name, spec, n_gpus in cfg.CLUSTER]
    events: list[dict] = []

    hot_blocks = requests[0].blocks[: ceil_blocks(shared_prefix_tokens, cfg.BLOCK_TOKENS)]
    signal_s = max(0.0, burst_start_s - policy.lead_s)
    scheduled = False
    pending: list[PendingWarm] = []
    warm_stats = {"warm_targets": 0, "warm_bytes": 0, "warm_busy_s": 0.0}
    ready_warms = 0
    ready_by_burst = 0

    for req in requests:
        if not scheduled and policy.warm_kind and signal_s <= req.arrival:
            for node in nodes:
                admit_and_decode_until(
                    node, signal_s, nodes, cfg, events, allow_remote=policy.allow_remote_on_admit
                )
            pending, ready = process_ready_warms(pending, signal_s)
            ready_warms += ready
            new_pending, warm_stats = schedule_warm(
                policy, signal_s, nodes, hot_blocks, shared_prefix_tokens
            )
            ready_by_burst += sum(1 for warm in new_pending if warm.ready <= burst_start_s + 1e-12)
            pending.extend(new_pending)
            scheduled = True

        for node in nodes:
            admit_and_decode_until(
                node, req.arrival, nodes, cfg, events, allow_remote=policy.allow_remote_on_admit
            )
        pending, ready = process_ready_warms(pending, req.arrival)
        ready_warms += ready

        chosen = choose_node(policy, req, nodes, hot_blocks, req.arrival)
        chosen.waiting.append(req)

    if policy.warm_kind and not scheduled:
        for node in nodes:
            admit_and_decode_until(
                node, signal_s, nodes, cfg, events, allow_remote=policy.allow_remote_on_admit
            )
        new_pending, warm_stats = schedule_warm(
            policy, signal_s, nodes, hot_blocks, shared_prefix_tokens
        )
        ready_by_burst += sum(1 for warm in new_pending if warm.ready <= burst_start_s + 1e-12)
        pending.extend(new_pending)

    while pending:
        next_ready = min(warm.ready for warm in pending)
        for node in nodes:
            admit_and_decode_until(
                node, next_ready, nodes, cfg, events, allow_remote=policy.allow_remote_on_admit
            )
        pending, ready = process_ready_warms(pending, next_ready)
        ready_warms += ready

    for node in nodes:
        admit_and_decode_until(node, None, nodes, cfg, events, allow_remote=policy.allow_remote_on_admit)

    events.sort(key=lambda e: e["id"])
    return summarize(policy, cfg, nodes, events, warm_stats, ready_warms, ready_by_burst, burst_start_s)


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)
    return ordered[idx]


def summarize(
    policy: Policy,
    cfg: SimpleNamespace,
    nodes: list[Node],
    events: list[dict],
    warm_stats: dict,
    ready_warms: int,
    ready_by_burst: int,
    burst_start_s: float,
) -> dict:
    hot = [event for event in events if event["id"] != 0 and event["arrival"] >= burst_start_s]
    ttft = [event["start"] - event["arrival"] + event["reuse"] + event["prefill"] for event in hot]
    queue = [event["start"] - event["arrival"] for event in hot]
    lat = [event["finish"] - event["arrival"] for event in hot]
    prefix_tok = sum(event["prefix_tokens"] for event in hot)
    hit_tok = sum(event["hit"] for event in hot)
    rdma_s = sum(event["reuse"] for event in hot if event["tier"] == "rdma")
    out_tok = sum(event["output_tokens"] for event in hot)
    span = max((event["finish"] for event in events), default=0.0)
    counts = {node.name: 0 for node in nodes}
    for event in hot:
        counts[event["node"]] += 1

    return {
        "policy": policy.name,
        "mean_ttft": statistics.fmean(ttft) if ttft else 0.0,
        "p95_ttft": pct(ttft, 0.95),
        "p99_ttft": pct(ttft, 0.99),
        "max_ttft": max(ttft) if ttft else 0.0,
        "mean_queue": statistics.fmean(queue) if queue else 0.0,
        "p95_queue": pct(queue, 0.95),
        "mean_lat": statistics.fmean(lat) if lat else 0.0,
        "p95_lat": pct(lat, 0.95),
        "cache_hit": hit_tok / prefix_tok if prefix_tok else 0.0,
        "critical_rdma_s": rdma_s,
        "warm_gb": warm_stats["warm_bytes"] / config.GB,
        "warm_busy_s": warm_stats["warm_busy_s"],
        "warm_targets": warm_stats["warm_targets"],
        "ready_warms": ready_warms,
        "ready_by_burst": ready_by_burst,
        "throughput": out_tok / max(1e-9, span - burst_start_s),
        "util": sum(node.busy for node in nodes) / max(1e-9, len(nodes) * span),
        "node_counts": counts,
    }


def policies_for(rdma_enabled: bool, lead_s: float) -> list[Policy]:
    policies = [
        Policy("cache_only_affinity", "cache_only", allow_remote_on_admit=False),
        Policy("least_load_recompute", "least_load", allow_remote_on_admit=False),
    ]
    if rdma_enabled:
        policies.extend(
            [
                Policy("reactive_copy_rdma", "least_load", allow_remote_on_admit=True),
                Policy(
                    "predict_copy_rep2",
                    "warmed_cache",
                    allow_remote_on_admit=False,
                    warm_kind="copy",
                    replicas=2,
                    lead_s=lead_s,
                ),
                Policy(
                    "predict_copy_rep4",
                    "warmed_cache",
                    allow_remote_on_admit=False,
                    warm_kind="copy",
                    replicas=4,
                    lead_s=lead_s,
                ),
            ]
        )
    policies.extend(
        [
            Policy(
                "predict_fake_prefill_rep2",
                "warmed_cache",
                allow_remote_on_admit=False,
                warm_kind="fake_prefill",
                replicas=2,
                lead_s=lead_s,
            ),
            Policy(
                "predict_fake_prefill_rep4",
                "warmed_cache",
                allow_remote_on_admit=False,
                warm_kind="fake_prefill",
                replicas=4,
                lead_s=lead_s,
            ),
        ]
    )
    return policies


def format_counts(counts: dict[str, int]) -> str:
    return "/".join(str(counts[name]) for name in sorted(counts))


def print_results(title: str, results: list[dict]) -> None:
    print(f"\n{title}")
    print(
        f"{'policy':<28} {'meanTTFT':>9} {'p95TTFT':>9} {'p99TTFT':>9} "
        f"{'p95Q':>8} {'hit':>6} {'rdmaCrit':>9} {'warmGB':>7} {'ready':>5} {'warmBusy':>8} {'split':>15}"
    )
    for row in results:
        print(
            f"{row['policy']:<28} "
            f"{row['mean_ttft']:>8.2f}s "
            f"{row['p95_ttft']:>8.2f}s "
            f"{row['p99_ttft']:>8.2f}s "
            f"{row['p95_queue']:>7.2f}s "
            f"{row['cache_hit']:>5.0%} "
            f"{row['critical_rdma_s']:>8.2f}s "
            f"{row['warm_gb']:>6.1f} "
            f"{row['ready_by_burst']:>5} "
            f"{row['warm_busy_s']:>7.2f}s "
            f"{format_counts(row['node_counts']):>15}"
        )


def print_copy_vs_prefill(cfg: SimpleNamespace, prefix_sizes: list[int]) -> None:
    node = Node("probe", cfg.CLUSTER[0][1], cfg.CLUSTER[0][2], cfg)
    print("\nWarm cost per additional replica")
    print(f"{'shared prefix':>13} {'KV bytes':>10} {'RDMA copy':>11} {'fake prefill':>13} {'prefill/copy':>13}")
    for tokens in prefix_sizes:
        blocks = ceil_blocks(tokens, cfg.BLOCK_TOKENS)
        actual_tokens = blocks * cfg.BLOCK_TOKENS
        nbytes = blocks * node.block_bytes
        copy_s = node.load_time(nbytes, "rdma") if node.tier_bw["rdma"] > 0 else math.inf
        prefill_s = node.prefill_time(actual_tokens)
        ratio = prefill_s / copy_s if math.isfinite(copy_s) and copy_s > 0 else math.inf
        copy_text = f"{copy_s:>10.3f}s" if math.isfinite(copy_s) else "unavailable"
        ratio_text = f"{ratio:>12.1f}x" if math.isfinite(ratio) else "n/a"
        print(
            f"{actual_tokens:>10} tok "
            f"{nbytes / config.GB:>9.2f}G "
            f"{copy_text:>11} "
            f"{prefill_s:>12.3f}s "
            f"{ratio_text:>13}"
        )


def run_scenario(args: argparse.Namespace, *, name: str, rdma_gbps: float) -> list[dict]:
    cfg = make_cfg(rdma_gbps)
    requests = build_hot_burst(
        cfg,
        shared_prefix_tokens=args.prefix_tokens,
        burst=args.burst,
        burst_start_s=args.burst_start,
        burst_window_s=args.burst_window,
        suffix_tokens=args.suffix_tokens,
        output_tokens=args.output_tokens,
    )
    results = [
        run_policy(
            policy,
            cfg,
            requests,
            shared_prefix_tokens=args.prefix_tokens,
            burst_start_s=args.burst_start,
        )
        for policy in policies_for(rdma_gbps > 0, args.lead)
    ]
    print_results(name, results)
    return results


def print_lead_sweep(args: argparse.Namespace, *, rdma_gbps: float) -> None:
    cfg = make_cfg(rdma_gbps)
    requests = build_hot_burst(
        cfg,
        shared_prefix_tokens=args.prefix_tokens,
        burst=args.burst,
        burst_start_s=args.burst_start,
        burst_window_s=args.burst_window,
        suffix_tokens=args.suffix_tokens,
        output_tokens=args.output_tokens,
    )
    action = "copy" if rdma_gbps > 0 else "fake_prefill"
    policy_name = "predict_copy_rep4" if rdma_gbps > 0 else "predict_fake_prefill_rep4"
    print(f"\nLead-time sweep ({policy_name})")
    print(f"{'lead':>7} {'ready@burst':>11} {'meanTTFT':>9} {'p95TTFT':>9} {'p95Q':>8} {'warmGB':>7} {'warmBusy':>8}")
    for lead in args.sweep_leads:
        policy = Policy(
            policy_name,
            "warmed_cache",
            warm_kind=action,
            replicas=4,
            lead_s=lead,
        )
        result = run_policy(
            policy,
            cfg,
            requests,
            shared_prefix_tokens=args.prefix_tokens,
            burst_start_s=args.burst_start,
        )
        print(
            f"{lead:>6.2f}s "
            f"{result['ready_by_burst']:>11} "
            f"{result['mean_ttft']:>8.2f}s "
            f"{result['p95_ttft']:>8.2f}s "
            f"{result['p95_queue']:>7.2f}s "
            f"{result['warm_gb']:>6.1f} "
            f"{result['warm_busy_s']:>7.2f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--burst", type=int, default=192)
    parser.add_argument("--burst-start", type=float, default=5.0)
    parser.add_argument("--burst-window", type=float, default=0.8)
    parser.add_argument("--suffix-tokens", type=int, default=256)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--lead", type=float, default=0.75)
    parser.add_argument(
        "--rdma-gbps",
        type=float,
        default=50.0,
        help="per-GPU RDMA bandwidth in GB/s for the normal scenario",
    )
    parser.add_argument(
        "--skip-no-rdma",
        action="store_true",
        help="only run the normal RDMA scenario",
    )
    parser.add_argument(
        "--sweep-leads",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "Hot-prefix burst setup: "
        f"{args.burst} requests, {args.prefix_tokens} shared prefix tokens, "
        f"{args.suffix_tokens} unique suffix tokens, {args.output_tokens} output tokens, "
        f"burst at t={args.burst_start:.1f}s over {args.burst_window:.1f}s."
    )

    rdma_results = run_scenario(
        args,
        name=f"Scenario A: RDMA available ({args.rdma_gbps:g} GB/s per GPU)",
        rdma_gbps=args.rdma_gbps,
    )
    cfg = make_cfg(args.rdma_gbps)
    print_copy_vs_prefill(cfg, [2048, 8192, 16384, 32768])
    print_lead_sweep(args, rdma_gbps=args.rdma_gbps)

    if not args.skip_no_rdma:
        run_scenario(
            args,
            name="Scenario B: no peer KV transfer path",
            rdma_gbps=0.0,
        )
        print_lead_sweep(args, rdma_gbps=0.0)


if __name__ == "__main__":
    main()
