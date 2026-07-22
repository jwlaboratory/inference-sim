# inference-sim

A small, composable simulator for LLM inference clusters. It replays a
window of a real production trace, routes each request to a node with a
pluggable policy, and models continuous batching, prefill/decode, and
prefix-cache timing from first principles. A React UI visualizes the run
and lets you edit every tunable.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python server.py          # serves the UI + API on :8000
```

Open http://localhost:8000. CLI instead: `python3 simulate.py` compares all
routing policies on the same trace window.

UI development: `cd ui && npm install && npm run dev` (Vite on :5173,
proxies /api to :8000). `npm run build` refreshes what server.py serves.

## How it works

```
config.py     every tunable, grouped by consumer; defaults for everything
workload.py    fetches a window of the HF trace -> list[Request]
router.py      heuristic policies + policy registry
gpu.py         Node (n GPUs serving together) + timing equations + KV prefix cache
simulate.py    the event loop: route -> admit (prefix reuse + prefill) -> batched decode
server.py      FastAPI: serves ui/dist and POST /api/simulate
rl/            learned router policy, CEM trainer, and saved weights
ui/            React (Vite) frontend: config editor, metrics, Gantt playback
```

Research-specific BTB experiments are intentionally kept out of this repo. In
the recommended local layout they live in the sibling
`../bite-the-bullet/experiments/` tree and import these simulator modules.

**Workload.** Requests come from a Mooncake-format trace
([ART-Chat-2.5M](https://huggingface.co/datasets/alessiotoniolo/ART-Chat-2.5M)):
each row has real arrival timestamps, input/output token counts, and
`hash_ids` — hashes of consecutive 256-token prompt blocks. A random point
in the 2.5M-request trace is chosen (reproducible via `SEED`, or pin
`DATASET_OFFSET`) and `NUM_REQUESTS` consecutive requests are replayed,
with arrival gaps scaled by `ARRIVAL_SCALE`. Rows stream from the HF
datasets-server API; nothing is downloaded up front.

**Nodes.** The cluster is a list of nodes, each `n` GPUs of one spec
serving together (tensor parallel): compute, HBM bandwidth/capacity, PCIe
and RDMA aggregate across the node's GPUs. Nodes are independent replicas
— each must hold the full model in its combined HBM (the run errors out
otherwise) — and only share prefix caches over RDMA.

**Continuous batching.** Each node decodes all admitted requests as one
batch: a decode step reads the active weights once for the whole batch
plus every sequence's KV cache, so throughput scales with batch size until
KV headroom (HBM left after weights) or `MAX_BATCH` is hit. Admission
(prefix load + prefill) briefly pauses decode, as in prefill-prioritizing
schedulers; requests wait in a per-node queue until they fit.

**Prefix caching.** Two requests share a prefix exactly when their leading
block hashes match, so caches operate on the recorded hashes directly. Each
node keeps hot blocks in HBM (LRU), evicting to host RAM; every computed
block is also persisted to disk when `DISK_CACHE` is on. For each request
the simulator finds the longest cached run of its leading blocks across
local HBM (free), local RAM (PCIe), a peer node (RDMA), or disk — and uses
it only if loading beats recomputing.

**Timing model** (per node; bandwidths/FLOPs below are node aggregates):

- prefill: `2 * active_params * tokens / (flops * MFU)` (compute-bound)
- decode step (whole batch, one token each):
  `(active_weight_bytes + batch_kv_bytes) / (hbm_bw * MBU)` (memory-bound)
- cache load: `blocks * block_kv_bytes / tier_bandwidth`

**Routing.** `cache_aware` routes to the longest prefix match and falls
back to least-load when the cluster is imbalanced (SGLang's
`balance_abs/rel_threshold` scheme). Load = requests in flight on a node
(decode batch + waiting queue).

## Configuring

Everything lives in `config.py` (the CLI reads it directly; the UI posts
overrides per run): trace window and arrival scaling, served model shape
(`PARAMS`, `ACTIVE_PARAMS`, `LAYERS`, `KV_HEADS`, `HEAD_DIM`, `MFU`,
`MBU`), serving (`MAX_BATCH`), router thresholds, and the cluster — a list
of nodes `(name, GPUSpec, gpu_count)` where a `GPUSpec` is FLOPs, HBM
bandwidth/capacity, PCIe, RDMA, disk. Add nodes or new specs in one line
each; the UI can also edit all of this live.
