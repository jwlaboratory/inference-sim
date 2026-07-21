"""Routing policies, modeled on the SGLang model gateway.

Each policy is constructed with the config namespace; route(request, gpus,
now) returns the chosen GPU. Load = seconds of work already queued.

CacheAware follows SGLang's scheme: route to the GPU with the longest
prefix-block match, but fall back to least-load whenever the cluster is
imbalanced (both thresholds exceeded, as in balance_abs/rel_threshold).
"""
import random


def load(gpu, now):
    return max(0.0, gpu.free_at - now)


class Random:
    def __init__(self, cfg=None):
        pass

    def route(self, req, gpus, now):
        return random.choice(gpus)


class RoundRobin:
    def __init__(self, cfg=None):
        self.i = 0

    def route(self, req, gpus, now):
        self.i += 1
        return gpus[self.i % len(gpus)]


class LeastLoad:
    def __init__(self, cfg=None):
        pass

    def route(self, req, gpus, now):
        return min(gpus, key=lambda g: load(g, now))


class CacheAware:
    def __init__(self, cfg):
        self.cfg = cfg

    def route(self, req, gpus, now):
        loads = [load(g, now) for g in gpus]
        if max(loads) > self.cfg.IMBALANCE_ABS \
                and max(loads) > self.cfg.IMBALANCE_REL * min(loads):
            return min(gpus, key=lambda g: load(g, now))
        # longest prefix-block match, ties broken by lightest load
        return max(gpus, key=lambda g: (sum(g.match(req.blocks)), -load(g, now)))


POLICIES = {"cache_aware": CacheAware, "least_load": LeastLoad,
            "round_robin": RoundRobin, "random": Random}
