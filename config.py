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
ARRIVAL_SCALE = 1.0         # multiply real inter-arrival gaps (>1 = calmer)
BLOCK_TOKENS = 256          # tokens per prefix-hash block (Mooncake block size)

# --------------------------------------------------------------------- gpu.py
# model being served (Llama 3.3 70B, fp16 weights, GQA) — must fit in every
# node's combined HBM (weights are sharded across a node's GPUs; the run
# errors out if any node is too small)
PARAMS = 70.6e9       # total weights (HBM footprint)
ACTIVE_PARAMS = 70.6e9  # weights touched per token; < PARAMS for MoE, else = PARAMS
DTYPE_BYTES = 2       # bytes per weight/KV element (2 fp16, 1 fp8/int8, 0.5 int4)
LAYERS, KV_HEADS, HEAD_DIM = 80, 8, 128

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


# flops = peak dense BF16 (no sparsity). ram_bw ~ realistic PCIe for the part;
# rdma_bw ~ per-GPU NIC (400G = 50 GB/s, 800G = 100 GB/s).
A100 = GPUSpec("A100", 312e12, 2.00e12, 80 * GB, 25e9, 25e9, 5e9)
H100 = GPUSpec("H100", 989e12, 3.35e12, 80 * GB, 55e9, 50e9, 7e9)
H200 = GPUSpec("H200", 989e12, 4.80e12, 141 * GB, 55e9, 50e9, 7e9)
B200 = GPUSpec("B200", 2250e12, 8.00e12, 192 * GB, 60e9, 100e9, 7e9)
B300 = GPUSpec("B300", 2250e12, 8.00e12, 288 * GB, 120e9, 100e9, 7e9)
MI300X = GPUSpec("MI300X", 1307e12, 5.30e12, 192 * GB, 55e9, 50e9, 7e9)
MI355X = GPUSpec("MI355X", 2500e12, 8.00e12, 288 * GB, 120e9, 100e9, 7e9)

# ------------------------------------------------------------------ router.py
# cache-aware falls back to least-load when both thresholds are exceeded.
# Load = requests in flight on a node (decode batch + waiting queue), as in
# the SGLang gateway; keep ABS around a typical batch's worth of requests.
IMBALANCE_ABS = 8      # in-flight requests
IMBALANCE_REL = 1.5    # max_load > REL * min_load

# ---------------------------------------------------------------- simulate.py
# cluster = list of nodes: (name, GPUSpec, gpu count). GPUs within a node
# serve together (tensor parallel: compute/bandwidth/HBM aggregate); nodes
# are independent replicas and only share prefix caches over RDMA.
CLUSTER = [("node0", H100, 4), ("node1", H100, 4),
           ("node2", H100, 4), ("node3", H100, 4)]
MAX_BATCH = 256     # max sequences decoding together on one node
DISK_CACHE = True   # every computed prefix block is also persisted to local NVMe


def as_dict():
    """All tunables as a dict — the sim modules take these as a namespace,
    so a server (or test) can override any of them per run."""
    return {k: v for k, v in globals().items() if k.isupper()}
