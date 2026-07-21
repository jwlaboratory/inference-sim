"""GPU model: hardware specs, the served model, and timing equations.

Units: seconds, bytes, tokens. Bandwidth in bytes/s, compute in FLOP/s.

Each GPU serves one request at a time (no continuous batching) and keeps a
KV prefix cache at block granularity (BLOCK_TOKENS tokens per block, keyed
by the trace's hash_ids): hot blocks in HBM with LRU eviction, evictees
offloaded to host RAM. Loading cached blocks from ram/rdma/disk costs
bytes/bandwidth.
"""
from collections import OrderedDict


class GPU:
    def __init__(self, name, spec, cfg):
        self.name, self.spec, self.cfg = name, spec, cfg
        self.weight_bytes = cfg.PARAMS * cfg.DTYPE_BYTES           # HBM footprint
        self.active_bytes = cfg.ACTIVE_PARAMS * cfg.DTYPE_BYTES   # read per token
        self.kv_per_tok = 2 * cfg.LAYERS * cfg.KV_HEADS * cfg.HEAD_DIM * cfg.DTYPE_BYTES
        self.block_bytes = self.kv_per_tok * cfg.BLOCK_TOKENS
        self.free_at = 0.0   # when the current queue drains
        self.busy = 0.0      # total seconds spent serving
        self.budget_blocks = int(max(0, spec.hbm_cap - self.weight_bytes) / self.block_bytes)
        self.hbm = OrderedDict()  # block hash -> True, LRU order
        self.ram = set()          # blocks evicted from HBM

    # --- timing equations ---
    def prefill_time(self, tokens):
        """Compute-bound: 2 FLOPs per active parameter per token."""
        return 2 * self.cfg.ACTIVE_PARAMS * tokens / (self.spec.flops * self.cfg.MFU)

    def decode_time(self, out_tokens, context_tokens):
        """Memory-bound: each token reads the active weights plus the KV cache."""
        per_token = (self.active_bytes + context_tokens * self.kv_per_tok) \
            / (self.spec.hbm_bw * self.cfg.MBU)
        return out_tokens * per_token

    def load_time(self, nbytes, tier):
        """Time to move bytes into HBM from a storage tier."""
        bw = {"hbm": self.spec.hbm_bw, "ram": self.spec.ram_bw,
              "rdma": self.spec.rdma_bw, "disk": self.spec.disk_bw}[tier]
        return nbytes / bw

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
