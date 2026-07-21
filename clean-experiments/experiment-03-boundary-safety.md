# Experiment 3: Boundary And Safety

Status: **drafted, not run**.

## Question

Does adaptive warming avoid harm on workloads where fixed warming should not be
used, especially decode-heavy generation and public trace replay?

## Hypothesis

Fixed warming is good for short-output fanout but can hurt decode-heavy
workloads by adding prefill/decode interference. Adaptive idle-only warming
should be safer: smaller upside on target bursts, but near-neutral or slightly
positive on decode-heavy and trace replay.

## Real-System Setup

- Cluster: 8 replicas.
- Replica: 8 x H100 80GB, `TP=8`.
- Total GPUs: 64 x H100.
- Model: GLM-5.2-like 744B MoE.
- Weight dtype: int4 or fp8.
- KV dtype: fp8.
- Active params/token: 40B.
- Context cap: 128k.
- Prefix cache block size:
  - 256 tokens for synthetic decode-heavy workload.
  - 512 tokens for Mooncake trace replay.
- Cache tier: HBM only for primary decode-heavy run.
- RDMA boundary: 50 GB/s/GPU for trace replay boundary.
- Max batch: 256.
- Cache-aware threshold: `abs=8`, `rel=1.5`.

## Simulator Proxy Setup

- Model: current simulator default Llama-70B proxy.
- Cluster: current simulator default 4 replicas x 4 H100.
- Cache-aware threshold: `abs=8`, `rel=1.5`.

## Workload A: Decode-Heavy Fanout

- True burst jobs: 8.
- Requests per true job: 500.
- Shared prefix per true job: 65,536 tokens.
- Unique suffix per request: 256 tokens.
- Output tokens per request: 256.
- Burst arrival window: 1 second.
- First burst start: 20 seconds.
- Burst spacing: 40 seconds.
- False-positive candidate jobs: 120.
- Requests per false-positive job: 1.
- Background requests: 0.
- Predictor recall: 1.0.
- Predictor precision sweep:
  - 1.0
  - approximately 0.75
  - 0.5
  - 0.25
- Signal lead time: 6 seconds.

## Workload B: Public Trace Replay

- Dataset: `valeriol29/mooncake-traces`.
- Config: `mooncake`.
- Split: default split from script.
- Windows: 12.
- Rows per window: 1500.
- Max parquet files: 1.
- Block tokens: 512.
- Prefix key blocks: 8.
- Arrival scale: 0.25 for stress replay.
- Predictor model: `bite-the-bullet/mooncake_burst_model.json`.
- Threshold: 0.8.

## Policies

For Workload A:

- `cache_aware_default`.
- `least_load_recompute`.
- `fixed_32768`: fixed 32k partial prefill.
- `fixed_65536`: full prefill.
- `adaptive_empty_full`: idle-only adaptive warming.

For Workload B:

- `cache_aware_no_remote`.
- `least_load_no_remote`.
- `predict_fake`.
- `seed_real`.
- `reactive_copy_rdma`, only for RDMA boundary.

## Primary Metrics

- p95 TTFT delta versus `cache_aware_default`.
- mean TTFT delta versus `cache_aware_default`.
- p95 end-to-end latency.
- warm GPU-seconds.
- warmed KV GB.
- foreground/background p95 harm.
- RDMA critical path time for trace replay boundary.

## Success Criteria

This experiment supports the safety claim if:

- on decode-heavy fanout, `adaptive_empty_full` has lower p95 harm than
  `fixed_32768` and `fixed_65536`;
- on decode-heavy fanout, fixed warming is allowed to lose, but adaptive should
  be near-neutral or better than cache-aware p95 TTFT;
- on Mooncake trace replay without large synthetic fanouts, predictive warming
  should be near-neutral or clearly explainable;
- with RDMA enabled, reactive RDMA may dominate; if so, we report this as a
  boundary condition.

## Commands After Approval

Decode-heavy adaptive run:

```bash
python3 partial-prefill/adaptive_partial_prefill.py \
  --imbalance-abs 8 \
  --num-bursts 8 \
  --burst-size 500 \
  --prefix-tokens 65536 \
  --suffix-tokens 256 \
  --output-tokens 256 \
  --first-burst-s 20 \
  --burst-spacing-s 40 \
  --burst-window-s 1 \
  --num-decoys 120 \
  --background-requests 0 \
  --lead-s 6 \
  --precision-sweep 1 0.75 0.5 0.25 \
  --trials 5 \
  --fixed-warm-tokens 32768 65536 \
  --out clean-experiments/results/experiment_03a_decode_heavy.json
```

Mooncake trace replay, no RDMA:

```bash
python3 bite-the-bullet/evaluate_art_warming.py \
  --model bite-the-bullet/mooncake_burst_model.json \
  --dataset valeriol29/mooncake-traces \
  --config-name mooncake \
  --block-tokens 512 \
  --key-blocks 8 \
  --windows 12 \
  --rows-per-window 1500 \
  --max-parquet-files 1 \
  --warm-blocks 8 \
  --thresholds 0.8 \
  --rdma-gbps 0 \
  --hbm-only \
  --include-real-seed \
  --arrival-scale 0.25 \
  --imbalance-abs 8 \
  --out clean-experiments/results/experiment_03b_mooncake_no_rdma.json
```

Mooncake trace replay, RDMA boundary:

```bash
python3 bite-the-bullet/evaluate_art_warming.py \
  --model bite-the-bullet/mooncake_burst_model.json \
  --dataset valeriol29/mooncake-traces \
  --config-name mooncake \
  --block-tokens 512 \
  --key-blocks 8 \
  --windows 12 \
  --rows-per-window 1500 \
  --max-parquet-files 1 \
  --warm-blocks 8 \
  --thresholds 0.8 \
  --rdma-gbps 50 \
  --include-fake-prefill \
  --include-real-seed \
  --arrival-scale 0.25 \
  --imbalance-abs 8 \
  --out clean-experiments/results/experiment_03c_mooncake_rdma.json
```

