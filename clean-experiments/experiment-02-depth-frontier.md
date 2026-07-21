# Experiment 2: Predictor And Depth Frontier

Status: **run on the simulator**. See
`results/experiment_02_depth_frontier.json` and `results/SUMMARY.md`.

## Question

When predictions are noisy, what prefix depth should we warm? Is partial
warming actually the robust point, or should we always fully prefill when a
signal fires?

## Hypothesis

Partial warming, especially 25-50% of the prefix, lies on the p95
TTFT-versus-warm-cost frontier. Full-prefix warming only wins when predictor
precision is very high; at moderate precision it should hurt p95.

## Real-System Setup

- Cluster: 8 replicas.
- Replica: 8 x H100 80GB, `TP=8`.
- Total GPUs: 64 x H100.
- Model: GLM-5.2-like 744B MoE.
- Weight dtype: int4 or fp8.
- KV dtype: fp8.
- Active params/token: 40B.
- Context cap: 128k.
- Prefix cache block size: 256 tokens.
- RDMA/remote KV: disabled.
- Cache tier: HBM only.
- Max batch: 256.
- Cache-aware threshold: `abs=8`, `rel=1.5`.

## Simulator Proxy Setup

- Model preset: `glm52-int4`.
- Model shape: 744B total params, 40B active params/token, 78 layers,
  compressed KV proxy `KV_HEADS=1`, `HEAD_DIM=288`, `DTYPE_BYTES=0.5`.
- Cluster: 8 replicas x 8 H100.
- RDMA: `0`.
- HBM-only: `true`.
- Cache-aware threshold: `abs=8`, `rel=1.5`.
- Max batch: 256.

## Workload

- True burst jobs: 8.
- Requests per true job: 500.
- Total true burst requests: 4000.
- Shared prefix per true job: 65,536 tokens.
- Unique suffix per request: 256 tokens.
- Output tokens per request: 1.
- Burst arrival window: 1 second.
- First burst start: 20 seconds.
- Burst spacing: 40 seconds.
- False-positive candidate jobs: 120.
- Requests per false-positive job: 1.
- False-positive shared prefix: 65,536 tokens.
- Background requests: 0.

## Predictor Signal

- Signal arrives: 6 seconds before predicted job start.
- Recall: fixed at 1.0 for the primary frontier.
- Precision sweep:
  - 1.0
  - approximately 0.75
  - 0.5
  - 0.25
- Precision is simulated by selecting true jobs and decoy jobs at the job-signal
  level, not per request.

## Warm Depth Sweep

- 2,048 tokens.
- 4,096 tokens.
- 8,192 tokens.
- 16,384 tokens.
- 32,768 tokens.
- 65,536 tokens.

## Policies

- `cache_aware_default`: no warming.
- `partial_prefill_2k`.
- `partial_prefill_4k`.
- `partial_prefill_8k`.
- `partial_prefill_16k`.
- `partial_prefill_32k`.
- `full_prefill_64k`.

## Primary Metrics

- p95 TTFT delta versus `cache_aware_default`.
- mean TTFT delta versus `cache_aware_default`.
- warm GPU-seconds.
- warmed KV GB.
- false-positive warm cost.
- p95 latency on decoy requests.
- Pareto frontier: p95 TTFT improvement versus warm GPU-seconds.

## Success Criteria

This experiment supports the mechanism claim if:

- at precision 0.5, `partial_prefill_16k` or `partial_prefill_32k` improves or
  matches `cache_aware_default` p95 TTFT;
- at precision 0.5, `full_prefill_64k` regresses p95 TTFT relative to
  `cache_aware_default` or is dominated by partial prefill on warm cost;
- 16k-32k warming is on the Pareto frontier across at least three precision
  points.

## Command After Approval

```bash
python3 partial-prefill/sweep_partial_prefill.py \
  --imbalance-abs 8 \
  --model-preset glm52-int4 \
  --num-replicas 8 \
  --gpus-per-replica 8 \
  --gpu H100 \
  --max-batch 256 \
  --block-tokens 256 \
  --num-bursts 8 \
  --burst-size 500 \
  --prefix-tokens 65536 \
  --suffix-tokens 256 \
  --output-tokens 1 \
  --first-burst-s 20 \
  --burst-spacing-s 40 \
  --burst-window-s 1 \
  --num-decoys 120 \
  --decoy-size 1 \
  --background-requests 0 \
  --lead-s 6 \
  --precision-sweep 1 0.75 0.5 0.25 \
  --recall 1 \
  --trials 5 \
  --warm-tokens 2048 4096 8192 16384 32768 65536 \
  --out clean-experiments/results/experiment_02_depth_frontier.json
```
