"""Node model: hardware specs, the served model, and timing equations.

Units: seconds, bytes, tokens. Bandwidth in bytes/s, compute in FLOP/s.

A Node is n_gpus GPUs of one spec serving together (tensor parallel): the
weights are sharded across them, so compute, HBM bandwidth and capacity
aggregate — as do PCIe and RDMA (per-GPU lanes and NICs); the node-local
NVMe pool is shared. Nodes are independent replicas: each must hold the
full model in its combined HBM (enforced here), and the router balances
requests across them.

Requests on a node decode as one continuous batch: every decode step reads
the active weights once for the whole batch plus each running sequence's
KV cache. Prefill (and prefix loading) briefly serializes with decode, as
in prefill-prioritizing schedulers. The batching loop itself lives in
simulate.advance; this class holds state and the timing equations.

Each node keeps a KV prefix cache at block granularity (BLOCK_TOKENS
tokens per block, keyed by the trace's hash_ids): hot blocks in HBM with
LRU eviction, evictees offloaded to host RAM. Loading cached blocks from
ram/rdma/disk costs bytes/bandwidth.
"""
from collections import OrderedDict, deque


class Node:
    def __init__(self, name, spec, n_gpus, cfg):
        self.name, self.spec, self.n_gpus, self.cfg = name, spec, int(n_gpus), cfg
        n = self.n_gpus
        self.flops = spec.flops * n
        self.hbm_bw = spec.hbm_bw * n
        self.hbm_cap = spec.hbm_cap * n
        self.tier_bw = {"hbm": spec.hbm_bw * n, "ram": spec.ram_bw * n,
                        "rdma": spec.rdma_bw * n, "disk": spec.disk_bw}

        self.weight_bytes = cfg.PARAMS * cfg.DTYPE_BYTES        # HBM footprint
        self.active_bytes = cfg.ACTIVE_PARAMS * cfg.DTYPE_BYTES  # read per token
        self.kv_per_tok = 2 * cfg.LAYERS * cfg.KV_HEADS * cfg.HEAD_DIM * cfg.DTYPE_BYTES
        self.block_bytes = self.kv_per_tok * cfg.BLOCK_TOKENS

        if self.weight_bytes > self.hbm_cap:
            raise ValueError(
                f"model does not fit: {self.weight_bytes / 1e9:.0f} GB of weights vs "
                f"{self.hbm_cap / 1e9:.0f} GB combined HBM on node '{name}' "
                f"({n}×{spec.name}) — add GPUs to the node or quantize")
        kv_free = self.hbm_cap - self.weight_bytes
        self.kv_budget = int(kv_free / self.kv_per_tok)  # tokens of KV alongside weights
        if self.kv_budget < cfg.BLOCK_TOKENS:
            raise ValueError(
                f"model fits on node '{name}' but leaves only {kv_free / 1e9:.1f} GB "
                f"({self.kv_budget} tokens) of KV headroom — add GPUs to the node or quantize")
        self.budget_blocks = int(kv_free / self.block_bytes)

        # continuous-batching state, advanced by simulate.advance
        self.now = 0.0          # node-local clock
        self.busy = 0.0         # wall seconds spent prefilling or decoding
        self.waiting = deque()  # routed requests not yet admitted to the batch
        self.running = []       # sequences currently decoding together

        self.hbm = OrderedDict()  # block hash -> True, LRU order
        self.ram = set()          # blocks evicted from HBM

    # --- timing equations ---
    def prefill_time(self, tokens):
        """Compute-bound: 2 FLOPs per active parameter per token."""
        return 2 * self.cfg.ACTIVE_PARAMS * tokens / (self.flops * self.cfg.MFU)

    def decode_segment(self, steps, batch, kv_tokens):
        """Wall time for `steps` decode iterations of a `batch`-wide batch
        holding `kv_tokens` of total context: each step reads the active
        weights once plus all KV, and every sequence grows one token per step."""
        weights = self.active_bytes * steps
        kv = self.kv_per_tok * (steps * kv_tokens + batch * steps * (steps - 1) / 2)
        return (weights + kv) / (self.hbm_bw * self.cfg.MBU)

    def load_time(self, nbytes, tier):
        """Time to move bytes into HBM from a storage tier."""
        return nbytes / self.tier_bw[tier]

    # --- prefix cache ---
    def match(self, blocks):
        """Longest leading run of blocks cached locally -> (in_hbm, in_ram)."""
        hbm_n = ram_n = 0
        for b in blocks:
            if b in self.hbm:
                hbm_n += 1
            elif b in self.ram:
                ram_n += 1
            else:
                break
        return hbm_n, ram_n

    def insert(self, blocks):
        for b in blocks:
            self.ram.discard(b)
            if b in self.hbm:
                self.hbm.move_to_end(b)
            else:
                self.hbm[b] = True
        while len(self.hbm) > self.budget_blocks:
            b, _ = self.hbm.popitem(last=False)   # LRU -> host RAM
            self.ram.add(b)
