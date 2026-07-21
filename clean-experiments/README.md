# Clean Experiments

Status: **run on the simulator**. See `results/SUMMARY.md`.

These are the three clean experiments. They are based on
what we learned from the exploratory `bite-the-bullet/` and `partial-prefill/`
work, but they are written as fresh paper-grade experiments with every setting
spelled out.

## Core Claim

Predictive partial-prefix KV warming is useful for metadata-predictable
shared-prefix bursts. The policy should use idle/slack capacity and usually
warm only the first part of the shared prefix, because full-prefix warming is
fragile under false positives and decode-heavy load.

This is not a claim that warming always beats cache-aware routing. It is a
claim about a specific workload class:

- data-labeling fanout;
- agent/subagent fanout;
- batch jobs with shared system/document/tool prefixes;
- jobs whose metadata reveals a near-future burst before all requests arrive.

## Common Real-System Setup

These settings apply to all three experiments unless an experiment overrides
them.

### Cluster

- Total serving replicas: 8.
- Replica shape: 1 node with 8 x H100 80GB SXM GPUs.
- Total GPUs: 64 x H100.
- Tensor parallelism: `TP=8` inside each replica.
- Data parallelism: 8 independent serving replicas.
- Intra-node interconnect: NVLink/NVSwitch.
- Inter-node network: disabled for primary HBM-only runs.
- RDMA boundary runs, when used: 50 GB/s/GPU.

### Model

Use a GLM-5.2-like MoE model shape.

- Total parameters: 744B.
- Active parameters per token: 40B.
- Layers: 78.
- KV layout: MLA-like compressed KV, modeled as `KV_HEADS=1`,
  `HEAD_DIM=288`.
- Real-system weight dtype: int4 or fp8.
- Real-system KV dtype: fp8 unless the serving stack only supports fp16 KV.
- Context cap for experiments: 128k tokens.
- Important fit note: bf16 744B weights do not fit on 8 x H100
  (`744B * 2 bytes = 1.49TB`, but 8 x H100 gives 640GB HBM). The real H100
  setup therefore requires quantized weights, or else the replica shape must be
  larger.

### Simulator Run Setup

The simulator now accepts the same GLM-like shape through
`--model-preset glm52-int4`. One caveat: it still uses one `DTYPE_BYTES` value
for both weights and KV, so this is an int4-weight/int4-KV stress proxy rather
than a fully faithful int4-weight/fp8-KV system.

- `--model-preset glm52-int4`
- `PARAMS = 744e9`
- `ACTIVE_PARAMS = 40e9`
- `LAYERS = 78`
- `KV_HEADS = 1`
- `HEAD_DIM = 288`
- `DTYPE_BYTES = 0.5`
- `--num-replicas 8`
- `--gpus-per-replica 8`
- `--gpu H100`

### Router And Cache

- Prefix cache block size: 256 tokens in synthetic simulator runs.
- Cache tier: HBM-only for primary experiments.
- RDMA: off unless explicitly enabled in Experiment 3.
- Cache-aware default:
  - `IMBALANCE_ABS = 8`
  - `IMBALANCE_REL = 1.5`
- Max batch: 256 sequences per replica.
- Continuous batching: enabled.

### Metrics

Record these for every experiment:

- TTFT mean, p50, p95, p99.
- End-to-end latency mean and p95.
- Queue delay mean and p95.
- Goodput under TTFT SLOs: 2s, 5s, 10s.
- Warm GPU-seconds.
- Warmed KV GB.
- Number of warm actions.
- Cache hit ratio.
- Burst split across replicas.
- Background/non-burst p95 latency, when background exists.

## The Three Experiments

1. [Experiment 1: Target Burst Win](experiment-01-target-burst.md)
   - Clean data-labeling fanout.
   - Perfect metadata predictor.
   - Shows the main positive result.

2. [Experiment 2: Predictor And Depth Frontier](experiment-02-depth-frontier.md)
   - Same burst family, but with decoys and predictor noise.
   - Sweeps warm depth and precision.
   - Shows why partial prefill is the robust mechanism.

3. [Experiment 3: Boundary And Safety](experiment-03-boundary-safety.md)
   - Decode-heavy fanout plus trace replay/RDMA boundary.
   - Shows when adaptive gating matters and when the idea should not claim a
     win.

## Approval Rule

Do not run these yet. After review, run all three in order and save outputs
under:

```text
clean-experiments/results/
```
