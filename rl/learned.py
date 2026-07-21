"""CEM-trained learned router policy for the simulator."""

PRED_OUT = 52.0   # decode-length guess; true output length is not observable


def _step_s(nd):
    """Per-decode-step weight-read time on this node."""
    return nd.active_bytes / (nd.hbm_bw * nd.cfg.MBU)


def features(req, nodes, momentum=0.0):
    """One observable feature vector per node for the linear learned router."""
    hits = [min(sum(nd.match(req.blocks)) * nd.cfg.BLOCK_TOKENS,
                req.prefix_tokens) for nd in nodes]
    out = []
    for i, nd in enumerate(nodes):
        wait_prefill = sum(nd.prefill_time(r.input_tokens) for r in nd.waiting)
        wait_decode = len(nd.waiting) * PRED_OUT * _step_s(nd)
        run_decode = sum(max(PRED_OUT - (s.req.output_tokens - s.left), 1.0)
                         for s in nd.running) * _step_s(nd)
        kv_used = sum(s.req.input_tokens + s.req.output_tokens
                      for s in nd.running) / nd.kv_budget
        remote = max((hits[j] for j in range(len(nodes)) if j != i), default=0)
        hit_frac = hits[i] / max(req.prefix_tokens, 1)
        out.append([wait_prefill,                                # 0 queued prefill s
                    wait_decode,                                 # 1 queued decode s
                    run_decode,                                  # 2 running decode s
                    len(nd.waiting) / 10, len(nd.running) / 10,  # 3,4 counts
                    nd.prefill_time(req.input_tokens - hits[i]), # 5 marginal prefill s
                    hit_frac,                                    # 6 local hit frac
                    remote / max(req.prefix_tokens, 1),          # 7 best remote frac
                    kv_used,                                     # 8 KV pressure
                    momentum,                                    # 9 group burst rate
                    momentum * hit_frac])                        # 10 burst x cache
    return out


MOMENTUM_WINDOW = 30.0   # seconds of same-group arrival history


class Learned:
    """Linear scoring router: route to the node minimizing W dot features.

    W was trained by cross-entropy-method search over trace-replay episodes in
    rl/rlgym.py: mean + p95 latency vs least_load, selected on held-out windows.
    """
    W = [1.116, 0.639, 1.620, -0.067, -0.066, 0.444,
         -0.090, 0.282, 0.198, -0.718, 0.732]   # CEM-trained 2026-07-21

    def __init__(self, cfg=None):
        self.group_times = {}   # group -> recent arrival times

    def momentum(self, req, now):
        ts = [t for t in self.group_times.get(req.group, [])
              if now - t <= MOMENTUM_WINDOW]
        ts.append(now)
        self.group_times[req.group] = ts
        return min(len(ts) - 1, 20) / 10

    def route(self, req, nodes, now):
        fs = features(req, nodes, self.momentum(req, now))
        scores = [sum(w * f for w, f in zip(self.W, fv)) for fv in fs]
        return nodes[scores.index(min(scores))]
