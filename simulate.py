"""Inference cluster simulator: replay the trace, route it, report.

For each request the chosen GPU reuses the longest cached run of its prefix
blocks — free from local HBM, else loaded from host RAM / a peer via RDMA /
disk, or simply recomputed if that's cheaper — then prefills the remaining
prompt tokens and decodes the output. run() returns metrics plus a
per-request event list (used by the web UI for playback). Run directly to
compare all routing policies on the same window.
"""
from types import SimpleNamespace

import config
from gpu import GPU
from router import POLICIES
from workload import generate


def default_cfg():
    return SimpleNamespace(**config.as_dict())


def prefix_source(req, gpu, gpus, disk, cfg):
    """Best cached copy of the request's leading blocks: (blocks, load_s, tier)."""
    hbm_n, ram_n = gpu.match(req.blocks)
    local = (hbm_n + ram_n, gpu.load_time(ram_n * gpu.block_bytes, "ram"),
             "hbm" if ram_n == 0 else "ram")
    candidates = [local]

    remote_n = max((sum(g.match(req.blocks)) for g in gpus if g is not gpu), default=0)
    candidates.append((remote_n, gpu.load_time(remote_n * gpu.block_bytes, "rdma"), "rdma"))

    if cfg.DISK_CACHE:
        disk_n = 0
        for b in req.blocks:
            if b not in disk:
                break
            disk_n += 1
        candidates.append((disk_n, gpu.load_time(disk_n * gpu.block_bytes, "disk"), "disk"))

    return max(candidates, key=lambda c: (c[0], -c[1]))


def run(policy, requests, cfg=None):
    cfg = cfg or default_cfg()
    gpus = [GPU(name, spec, cfg) for name, spec in cfg.CLUSTER]
    router = POLICIES[policy](cfg)
    disk = set()
    events, hit_tok, hbm_tok, prefix_tok = [], 0, 0, 0

    for r in requests:
        gpu = router.route(r, gpus, r.arrival)

        n, load, tier = prefix_source(r, gpu, gpus, disk, cfg)
        hit = min(n * cfg.BLOCK_TOKENS, r.prefix_tokens)
        reuse = min(load, gpu.prefill_time(hit))    # recompute if loading is slower
        prefill = gpu.prefill_time(r.input_tokens - hit)
        decode = gpu.decode_time(r.output_tokens, r.input_tokens)

        start = max(r.arrival, gpu.free_at)
        gpu.free_at = start + reuse + prefill + decode
        gpu.busy += reuse + prefill + decode

        gpu.insert(r.cache_blocks)                  # request context now resident
        if cfg.DISK_CACHE:
            disk.update(r.cache_blocks)

        hit_tok += hit
        hbm_tok += hit if tier == "hbm" else 0
        prefix_tok += r.prefix_tokens
        events.append({"id": r.id, "arrival": r.arrival, "group": r.group,
                       "prefix_tokens": r.prefix_tokens,
                       "input_tokens": r.input_tokens, "output_tokens": r.output_tokens,
                       "gpu": gpu.name, "start": start, "finish": gpu.free_at,
                       "reuse": reuse, "prefill": prefill, "decode": decode,
                       "hit": hit, "tier": tier if hit else "miss"})

    lat = sorted(e["finish"] - e["arrival"] for e in events)
    ttft = [e["start"] - e["arrival"] + e["reuse"] + e["prefill"] for e in events]
    span = max(g.free_at for g in gpus)
    return {"span": span,
            "gpus": [{"name": g.name, "budget": g.budget_blocks * cfg.BLOCK_TOKENS} for g in gpus],
            "events": events,
            "metrics": {"mean_lat": sum(lat) / len(lat),
                        "p95_lat": lat[int(0.95 * len(lat))],
                        "mean_ttft": sum(ttft) / len(ttft),
                        "cache_hit": hit_tok / prefix_tok if prefix_tok else 0,
                        "hbm_hit": hbm_tok / prefix_tok if prefix_tok else 0,
                        "util": sum(g.busy for g in gpus) / (len(gpus) * span)}}


if __name__ == "__main__":
    cfg = default_cfg()
    requests = generate(cfg)
    print(f"{len(requests)} requests on {len(cfg.CLUSTER)} GPUs "
          f"({', '.join(s.name for _, s in cfg.CLUSTER)})\n")
    print(f"{'policy':<12} {'mean_lat':>9} {'p95_lat':>9} {'mean_ttft':>10} "
          f"{'cache_hit':>10} {'hbm_hit':>8} {'util':>6}")
    for policy in POLICIES:
        m = run(policy, requests, cfg)["metrics"]
        print(f"{policy:<12} {m['mean_lat']:>8.1f}s {m['p95_lat']:>8.1f}s "
              f"{m['mean_ttft']:>9.1f}s {m['cache_hit']:>9.0%} "
              f"{m['hbm_hit']:>7.0%} {m['util']:>6.0%}")
