"""All tunables in one place, grouped by the module that uses them.

Units: seconds, bytes, tokens. Bandwidth in bytes/s, compute in FLOP/s.
"""
from dataclasses import dataclass

GB = 1e9

# ---------------------------------------------------------------- workload.py
# The workload replays a window of a real production trace from Hugging Face
# (Mooncake FAST'25 format: timestamp_ms, input_length, output_length, and
# hash_ids — hashes of 256-token prompt blocks). A random point in the trace
# is chosen and NUM_REQUESTS consecutive requests are replayed; the prefix
# caches match directly on the recorded block hashes.
SEED = 42
NUM_REQUESTS = 400          # window length: how many consecutive requests
DATASET = "alessiotoniolo/ART-Chat-2.5M"
DATASET_SPLIT = "train"
DATASET_OFFSET = -1         # start row; -1 = random point (derived from SEED)
ARRIVAL_SCALE = 4.0         # multiply real inter-arrival gaps (>1 = calmer)
BLOCK_TOKENS = 256          # tokens per prefix-hash block (Mooncake block size)

# --------------------------------------------------------------------- gpu.py
# model being served (8B-class, fp16 weights, GQA) — must fit in one GPU's HBM
PARAMS = 8e9
DTYPE_BYTES = 2
LAYERS, KV_HEADS, HEAD_DIM = 32, 8, 128

MFU = 0.5   # achieved fraction of peak compute during prefill
MBU = 0.8   # achieved fraction of peak HBM bandwidth during decode


@dataclass(frozen=True)
class GPUSpec:
    name: str
    flops: float     # peak dense FLOP/s
    hbm_bw: float    # HBM bandwidth
    hbm_cap: float   # HBM capacity
    ram_bw: float    # host RAM <-> GPU (PCIe)
    rdma_bw: float   # peer GPU over the network
    disk_bw: float   # local NVMe read


H100 = GPUSpec("H100", 989e12, 3.35e12, 80 * GB, 55e9, 50e9, 7e9)
A100 = GPUSpec("A100", 312e12, 2.00e12, 80 * GB, 25e9, 25e9, 5e9)

# ------------------------------------------------------------------ router.py
# cache-aware falls back to least-load when both thresholds are exceeded
IMBALANCE_ABS = 15.0   # seconds of queued work
IMBALANCE_REL = 1.5    # max_load > REL * min_load

# ---------------------------------------------------------------- simulate.py
CLUSTER = [("gpu0", H100), ("gpu1", H100), ("gpu2", H100), ("gpu3", H100)]
DISK_CACHE = True   # every computed prefix block is also persisted to local NVMe


def as_dict():
    """All tunables as a dict — the sim modules take these as a namespace,
    so a server (or test) can override any of them per run."""
    return {k: v for k, v in globals().items() if k.isupper()}
