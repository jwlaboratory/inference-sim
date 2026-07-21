# Clean Experiment Results

Run date: 2026-07-21.

## Setup Used

- Cluster: 8 independent replicas, each replica is 8 x H100 80GB with TP=8.
- Total GPUs: 64 x H100.
- Model preset: `glm52-int4`.
- Simulator model shape: 744B total params, 40B active params/token, 78 layers,
  MLA-like compressed KV proxy with `KV_HEADS=1`, `HEAD_DIM=288`.
- Simulator dtype note: the current simulator has one `DTYPE_BYTES` knob shared
  by weights and KV. `glm52-int4` sets it to 0.5 bytes, so these runs are an
  int4-weight/int4-KV proxy. Real fp8 KV would make warmed-KV bytes about 2x
  larger than these numbers.
- Max serving batch: `MAX_BATCH=256`.
- Synthetic block size: 256 tokens.
- Mooncake replay block size: 512 tokens.
- Cache-aware threshold: `IMBALANCE_ABS=8`, `IMBALANCE_REL=1.5`.

The short-output synthetic runs use `output_tokens=1` on purpose. They isolate
prefill/TTFT behavior and model labeling, scoring, extraction, and agent fanout
where the completion is tiny. Experiment 3A uses `output_tokens=256` to test the
decode-heavy boundary.

## Experiment 1: Target Burst Win

Workload: 8 bursts, 500 requests/burst, 65,536-token shared prefix,
256-token per-request suffix, 1 output token, no decoys, no RDMA.

Baseline p95 TTFT:

- `pure_cache_affinity`: 2.861s.
- `cache_aware_no_warm`: 1.313s.
- `least_load_recompute`: 1.298s.

Predictive warming versus `cache_aware_no_warm`:

| Warm depth | p95 TTFT | Delta p95 | Warm KV | Warm busy |
| --- | ---: | ---: | ---: | ---: |
| 16,384 tokens | 0.982s | -0.331s | 11.8 GB | 10.6s |
| 32,768 tokens | 0.651s | -0.662s | 23.6 GB | 21.2s |
| 65,536 tokens | 0.005s | -1.308s | 47.1 GB | 42.4s |

Result: the idea clearly wins on the target synthetic burst. Partial warming
buys a large p95 TTFT reduction at lower cost; full warming is best when the
predictor is perfect and idle warm time exists.

## Experiment 2: Predictor And Depth Frontier

Workload: same burst shape as Experiment 1, plus 120 one-request decoy jobs.
Precision sweep: 1.0, about 0.75, 0.5, 0.25.

Baseline p95 TTFT:

- `cache_aware_no_warm`: 1.728s.
- `least_load_recompute`: 1.731s.

Selected frontier points versus `cache_aware_no_warm`:

| Warm depth | Actual precision | p95 TTFT | Delta p95 | Warm KV | Warm busy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32,768 tokens | 1.00 | 1.176s | -0.551s | 23.6 GB | 21.2s |
| 32,768 tokens | 0.50 | 1.176s | -0.552s | 47.1 GB | 42.4s |
| 32,768 tokens | 0.25 | 1.076s | -0.652s | 94.2 GB | 84.8s |
| 65,536 tokens | 1.00 | 0.194s | -1.533s | 47.1 GB | 42.4s |
| 65,536 tokens | 0.50 | 0.176s | -1.552s | 94.2 GB | 84.8s |
| 65,536 tokens | 0.25 | 0.199s | -1.528s | 188.4 GB | 169.6s |

Result: under this GLM-like 8-replica setup, full warming remains the latency
winner even with these decoys. Partial warming is the lower-cost point, but this
run does not validate the stronger hypothesis that full warming becomes fragile
at moderate precision. The earlier intuition likely needs a harsher false
positive/background workload, fp8/fp16 KV cost modeling, tighter HBM budget, or
less idle warm slack.

## Experiment 3: Boundary And Safety

### 3A: Decode-Heavy Synthetic

Workload: same burst shape as Experiment 2, but 256 output tokens/request.

Baseline p95 TTFT:

- `pure_cache_affinity`: 9.677s.
- `cache_aware_no_warm`: 1.728s.
- `least_load_recompute`: 1.731s.

Selected policies:

| Policy | Precision | p95 TTFT | Warm KV | Warm busy |
| --- | ---: | ---: | ---: | ---: |
| `adaptive_empty_full` | 1.00 | 0.622s | 47.1 GB | 42.4s |
| `fixed_32768` | 1.00 | 1.284s | 23.6 GB | 21.2s |
| `fixed_65536` | 1.00 | 0.622s | 47.1 GB | 42.4s |
| `adaptive_empty_full` | 0.50 | 0.587s | 93.6 GB | 84.3s |
| `fixed_65536` | 0.50 | 0.587s | 94.2 GB | 84.8s |
| `adaptive_empty_conf` | 0.50 | 9.015s | 46.8 GB | 42.1s |

Result: decode-heavy load did not make fixed full warming lose in this roomy
setup. `adaptive_empty_full` ties fixed full warming while skipping a little
work when busy. Confidence-scaled warming is unsafe here because it underwarms
the true burst.

### 3B: Mooncake Replay, No RDMA

Workload: 12 windows x 1500 rows from `valeriol29/mooncake-traces`, arrival
scale 0.25, HBM-only, no RDMA.

Baseline is `least_load_no_remote`.

| Policy | p95 TTFT | Delta p95 | Warm KV |
| --- | ---: | ---: | ---: |
| `cache_aware_no_remote` | 0.768s | +0.006s | 0.00 GB |
| `least_load_no_remote` | 0.762s | 0.000s | 0.00 GB |
| `predict_fake_b8_t0.80` | 0.795s | +0.033s | 0.28 GB |
| `seed_real_b8_t0.80` | 0.794s | +0.032s | 0.18 GB |

Result: on this public trace slice, predictive warming is near-neutral but not
better. The trace does not produce the large same-prefix fanout that the
synthetic target workload models.

### 3C: Mooncake Replay, RDMA Boundary

Workload: same Mooncake replay, but with 50 GB/s/GPU RDMA enabled.

Baseline is `reactive_copy_rdma`.

| Policy | p95 TTFT | Delta p95 | Warm KV |
| --- | ---: | ---: | ---: |
| `reactive_copy_rdma` | 0.695s | 0.000s | 0.00 GB |
| `prewarm_reactive_copy_b8_t0.80` | 0.695s | 0.000s | 0.28 GB |
| `predict_fake_b8_t0.80` | 0.795s | +0.100s | 0.28 GB |
| `seed_real_b8_t0.80` | 0.794s | +0.098s | 0.18 GB |

Result: reactive RDMA is the ceiling on this trace. Predictive warming only
ties it when it still falls back to reactive copy on admission.

## Bottom Line

The idea is valid for a specific regime: predictable, synchronized,
same-prefix fanout with short outputs and enough idle warm lead time. It is not
a universal replacement for cache-aware routing or reactive RDMA.

The current clean runs say:

- yes, predictive warming beats cache-aware on the target synthetic burst;
- partial warming is a useful cost/performance point;
- full warming is best in the current GLM-like 8x8 H100 proxy when predictor
  signals are clean enough and the cluster has slack;
- Mooncake trace replay does not validate the target workload by itself;
- RDMA reactive copy remains a strong baseline and should be treated as a
  boundary condition.

Next simulator improvement: split weight dtype and KV dtype so GLM-style int4
weights with fp8 KV can be modeled directly.
