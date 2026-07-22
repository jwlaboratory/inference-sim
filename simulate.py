"""Inference cluster simulator: replay the trace, route it, report.

Each request is routed to a node on arrival and joins that node's waiting
queue. A node admits requests into its continuous batch whenever there is
KV headroom and the batch is under MAX_BATCH: admission reuses the longest
cached run of prefix blocks — free from local HBM, else loaded from host
RAM / an optional shared disk tier, or recomputed if that's cheaper — then
prefills the remaining prompt, briefly pausing decode (prefill-priority
scheduling, as in vLLM/SGLang defaults). All running sequences decode
together: each step reads the weights once for the whole batch, so decode
throughput scales with batch size until KV headroom runs out.

run() returns metrics plus a per-request event list (used by the web UI
for playback). Run directly to compare all routing policies on the same
trace window.
"""
from types import SimpleNamespace

import config
from gpu import Node
from router import POLICIES
from workload import generate


def default_cfg():
    return SimpleNamespace(**config.as_dict())


def prefix_source(req, node, disk, cfg):
    """Best cached copy of the request's leading blocks: (blocks, load_s, tier).

    A prefix the node holds locally is reused from HBM/RAM; otherwise it is
    recomputed, unless an optional shared disk tier is enabled to supply it."""
    hbm_n, ram_n = node.match(req.blocks)
    local = (hbm_n + ram_n, node.load_time(ram_n * node.block_bytes, "ram"),
             "hbm" if ram_n == 0 else "ram")
    candidates = [local]

    if cfg.DISK_CACHE:
        disk_n = 0
        for b in req.blocks:
            if b not in disk:
                break
            disk_n += 1
        candidates.append((disk_n, node.load_time(disk_n * node.block_bytes, "disk"), "disk"))

    return max(candidates, key=lambda c: (c[0], -c[1]))


class Seq:
    """A request admitted to a node's decode batch."""
    __slots__ = ("req", "start", "reuse", "prefill", "hit", "tier", "left", "context")

    def __init__(self, req, start, reuse, prefill, hit, tier):
        self.req, self.start = req, start
        self.reuse, self.prefill, self.hit, self.tier = reuse, prefill, hit, tier
        self.left = req.output_tokens        # decode tokens still to generate
        self.context = req.input_tokens      # KV tokens resident, grows as it decodes


def _finish(seq, node):
    r = seq.req
    return {"id": r.id, "arrival": r.arrival, "group": r.group,
            "prefix_tokens": r.prefix_tokens,
            "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
            "node": node.name, "start": seq.start, "finish": node.now,
            "reuse": seq.reuse, "prefill": seq.prefill,
            "decode": node.now - seq.start - seq.reuse - seq.prefill,
            "hit": seq.hit, "tier": seq.tier if seq.hit else "miss"}


def advance(node, until, nodes, disk, cfg, events):
    """Run one node's continuous-batching loop up to `until` (None = drain).

    Admissions (prefix load + prefill) serialize with decode and run to
    completion even if they overshoot `until` slightly; decode advances in
    closed-form segments that stop at the next completion or at `until`.
    """
    while True:
        # admit while there's KV headroom and a batch slot
        while node.waiting and len(node.running) < cfg.MAX_BATCH:
            req = node.waiting[0]
            need = req.input_tokens + req.output_tokens   # KV reserved up front
            used = sum(s.req.input_tokens + s.req.output_tokens for s in node.running)
            if used + need > node.kv_budget:
                if not node.running:
                    raise ValueError(
                        f"request {req.id} needs {need} tokens of KV but node "
                        f"'{node.name}' has headroom for {node.kv_budget} after "
                        f"weights — add GPUs to the node or quantize")
                break
            node.waiting.popleft()
            n, load, tier = prefix_source(req, node, disk, cfg)
            hit = min(n * cfg.BLOCK_TOKENS, req.prefix_tokens)
            recompute = node.prefill_time(hit)
            reuse = min(load, recompute)   # recompute if loading is slower
            prefill = node.prefill_time(req.input_tokens - hit)
            node.running.append(Seq(req, node.now, reuse, prefill, hit, tier))
            node.now += reuse + prefill
            node.busy += reuse + prefill
            node.insert(req.cache_blocks)               # request context now resident
            if cfg.DISK_CACHE:
                disk.update(req.cache_blocks)

        if until is not None and node.now >= until:
            return
        if not node.running:
            if until is not None:
                node.now = max(node.now, until)
            return

        # decode to the next completion, bounded by `until`
        batch = len(node.running)
        kv_tokens = sum(s.context for s in node.running)
        steps = min(s.left for s in node.running)
        if until is not None and node.decode_segment(steps, batch, kv_tokens) > until - node.now:
            lo, hi = 1, steps    # largest step count that fits, but always >= 1
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
        for s in node.running:
            s.left -= steps
            s.context += steps
        events.extend(_finish(s, node) for s in node.running if s.left == 0)
        node.running = [s for s in node.running if s.left > 0]


def run(policy, requests, cfg=None):
    cfg = cfg or default_cfg()
    nodes = [Node(name, spec, n, cfg) for name, spec, n in cfg.CLUSTER]
    router = POLICIES[policy](cfg)
    disk = set()
    events = []

    for r in requests:
        for node in nodes:
            advance(node, r.arrival, nodes, disk, cfg, events)
        router.route(r, nodes, r.arrival).waiting.append(r)
    for node in nodes:
        advance(node, None, nodes, disk, cfg, events)
    events.sort(key=lambda e: e["id"])

    hit_tok = sum(e["hit"] for e in events)
    hbm_tok = sum(e["hit"] for e in events if e["tier"] == "hbm")
    prefix_tok = sum(e["prefix_tokens"] for e in events)
    out_tok = sum(e["output_tokens"] for e in events)
    lat = sorted(e["finish"] - e["arrival"] for e in events)
    ttft = [e["start"] - e["arrival"] + e["reuse"] + e["prefill"] for e in events]
    peak_queue = max(e["start"] - e["arrival"] for e in events)
    span = max(e["finish"] for e in events)
    return {"span": span,
            "nodes": [{"name": nd.name, "spec": nd.spec.name, "gpus": nd.n_gpus,
                       "budget": nd.budget_blocks * cfg.BLOCK_TOKENS,
                       "kv_budget": nd.kv_budget} for nd in nodes],
            "events": events,
            "metrics": {"mean_lat": sum(lat) / len(lat),
                        "p50_lat": lat[len(lat) // 2],
                        "p95_lat": lat[int(0.95 * len(lat))],
                        "max_lat": lat[-1],
                        "mean_ttft": sum(ttft) / len(ttft),
                        "peak_queue": peak_queue,
                        "mean_queue": sum(e["start"] - e["arrival"] for e in events) / len(events),
                        "mean_prefill": sum(e["reuse"] + e["prefill"] for e in events) / len(events),
                        "mean_decode": sum(e["decode"] for e in events) / len(events),
                        "throughput": out_tok / span,
                        "cache_hit": hit_tok / prefix_tok if prefix_tok else 0,
                        "hbm_hit": hbm_tok / prefix_tok if prefix_tok else 0,
                        "util": sum(nd.busy for nd in nodes) / (len(nodes) * span)}}


if __name__ == "__main__":
    cfg = default_cfg()
    requests = generate(cfg)
    print(f"{len(requests)} requests on {len(cfg.CLUSTER)} nodes "
          f"({', '.join(f'{n}×{s.name}' for _, s, n in cfg.CLUSTER)})\n")
    print(f"{'policy':<12} {'mean_lat':>9} {'p95_lat':>9} {'mean_ttft':>10} "
          f"{'peak_q':>8} {'tok/s':>8} {'cache_hit':>10} {'hbm_hit':>8} {'util':>6}")
    for policy in POLICIES:
        m = run(policy, requests, cfg)["metrics"]
        print(f"{policy:<12} {m['mean_lat']:>8.1f}s {m['p95_lat']:>8.1f}s "
              f"{m['mean_ttft']:>9.1f}s {m['peak_queue']:>7.1f}s {m['throughput']:>8.0f} "
              f"{m['cache_hit']:>9.0%} {m['hbm_hit']:>7.0%} {m['util']:>6.0%}")
