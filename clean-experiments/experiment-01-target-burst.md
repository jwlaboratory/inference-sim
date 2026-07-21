# Experiment 1: Target Burst Win

Status: **drafted, not run**.

## Question

Can predictive partial-prefix warming reduce TTFT on the workload we actually
care about: a metadata-predictable data-labeling or agent fanout where hundreds
of requests share a long prefix?

## Hypothesis

With a clean predictor signal, warming the first 25-50% of the shared prefix on
extra replicas before the burst will beat default cache-aware routing on p95
TTFT and use less warm compute than full-prefix warming.

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
- RDMA/remote KV: disabled for the primary run.
- Cache tier: HBM only.
- Max batch: 256.
- Cache-aware threshold: `abs=8`, `rel=1.5`.

## Simulator Proxy Setup

- Model: current simulator default Llama-70B proxy.
- Cluster: current simulator default 4 replicas x 4 H100.
- RDMA: `0`.
- HBM-only: `true`.
- Cache-aware threshold: `abs=8`, `rel=1.5`.

## Workload

- Number of burst jobs: 8.
- Requests per job: 500.
- Total burst requests: 4000.
- Shared prefix per job: 65,536 tokens.
- Unique suffix per request: 256 tokens.
- Output tokens per request: 1.
- Burst arrival window: 1 second.
- First burst start: 20 seconds.
- Burst spacing: 40 seconds.
- Background requests: 0 for the clean primary run.
- Decoy/false-positive jobs: 0 for this experiment.

## Predictor Signal

- Signal arrives: 6 seconds before each burst.
- Predictor precision: 1.0.
- Predictor recall: 1.0.
- Signal source assumption: job metadata, batch id, tenant job id, or agent
  phase id.

## Policies

- `pure_cache_affinity`: always route to max prefix locality.
- `cache_aware_default`: cache-aware with `abs=8`, `rel=1.5`.
- `least_load_recompute`: load-only baseline.
- `partial_prefill_16k`: fake-prefill first 16,384 tokens.
- `partial_prefill_32k`: fake-prefill first 32,768 tokens.
- `full_prefill_64k`: fake-prefill full 65,536 token prefix.
- `adaptive_idle_partial`: warm only idle replicas, up to full prefix if time
  permits.

## Primary Metrics

- p95 TTFT on burst requests.
- mean TTFT on burst requests.
- p95 end-to-end latency on burst requests.
- warm GPU-seconds.
- warmed KV GB.
- burst split across replicas.

## Success Criteria

This experiment supports the main claim if:

- `partial_prefill_32k` improves p95 TTFT by at least 15% versus
  `cache_aware_default`;
- `partial_prefill_32k` uses no more than 60% of the warm compute of
  `full_prefill_64k`;
- `full_prefill_64k` is not strictly better than `partial_prefill_32k` on both
  p95 TTFT and warm compute.

## Command After Approval

```bash
python3 partial-prefill/sweep_partial_prefill.py \
  --imbalance-abs 8 \
  --num-bursts 8 \
  --burst-size 500 \
  --prefix-tokens 65536 \
  --suffix-tokens 256 \
  --output-tokens 1 \
  --first-burst-s 20 \
  --burst-spacing-s 40 \
  --burst-window-s 1 \
  --num-decoys 0 \
  --background-requests 0 \
  --lead-s 6 \
  --precision-sweep 1 \
  --recall 1 \
  --trials 5 \
  --warm-tokens 16384 32768 65536 \
  --out clean-experiments/results/experiment_01_target_burst.json
```

