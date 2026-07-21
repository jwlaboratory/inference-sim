"""Routing policies, modeled on the SGLang model gateway.

Each policy is constructed with the config namespace; route(request, nodes,
now) returns the chosen node. Load = requests in flight on the node (decode
batch + waiting queue), as in the SGLang gateway.

CacheAware follows SGLang's scheme: route to the node with the longest
prefix-block match, but fall back to least-load whenever the cluster is
imbalanced (both thresholds exceeded, as in balance_abs/rel_threshold).
"""
import random

from rl.learned import Learned


def load(node, now):
    return len(node.running) + len(node.waiting)


def pick(nodes, key):
    """Node minimizing key, ties broken randomly — with batching, loads (and
    prefix matches) are small ints that tie constantly, and min()'s
    first-element bias would pile every tie onto node0."""
    best = min(key(nd) for nd in nodes)
    return random.choice([nd for nd in nodes if key(nd) == best])


class Random:
    def __init__(self, cfg=None):
        pass

    def route(self, req, nodes, now):
        return random.choice(nodes)


class RoundRobin:
    def __init__(self, cfg=None):
        self.i = 0

    def route(self, req, nodes, now):
        self.i += 1
        return nodes[self.i % len(nodes)]


class LeastLoad:
    def __init__(self, cfg=None):
        pass

    def route(self, req, nodes, now):
        return pick(nodes, lambda nd: load(nd, now))


class CacheAware:
    def __init__(self, cfg):
        self.cfg = cfg

    def route(self, req, nodes, now):
        loads = [load(nd, now) for nd in nodes]
        if max(loads) > self.cfg.IMBALANCE_ABS \
                and max(loads) > self.cfg.IMBALANCE_REL * min(loads):
            return pick(nodes, lambda nd: load(nd, now))
        # longest prefix-block match, then lightest load, then random
        return pick(nodes, lambda nd: (-sum(nd.match(req.blocks)), load(nd, now)))


POLICIES = {"cache_aware": CacheAware, "least_load": LeastLoad,
            "round_robin": RoundRobin, "random": Random, "learned": Learned}
