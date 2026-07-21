# Standard-System Matrix

Run date: 2026-07-21.

All rows use the same target workload:

- 8 burst jobs.
- 500 requests per burst.
- 65,536-token shared prefix.
- 256-token unique suffix.
- 1 output token.
- 1 second burst arrival window.
- no false positives.
- no background traffic.
- no RDMA.
- HBM-only cache.
- max serving batch: 256.
- cache-aware threshold: `IMBALANCE_ABS=8`, `IMBALANCE_REL=1.5`.

Percentages are relative to `cache_aware_no_warm` p95 TTFT. Positive means
p95 TTFT reduction; negative means worse.

| System | Lead | Cache-aware p95 | Best warm depth | Warm p95 | p95 reduction | Mean TTFT reduction | Warm KV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-70B fp16, 4 replicas x 4 H100 | 6s | 6.333s | 65,536 | 1.638s | 74.1% | 84.4% | 687.2 GB |
| Llama-70B fp16, 8 replicas x 4 H100 | 6s | 5.047s | 65,536 | 1.638s | 67.5% | 82.2% | 687.2 GB |
| Llama-70B fp16, 8 replicas x 8 H100 | 6s | 2.339s | 65,536 | 0.290s | 87.6% | 93.3% | 687.2 GB |
| GLM-like 744B MoE int4 proxy, 8 replicas x 8 H100 | 6s | 1.313s | 65,536 | 0.005s | 99.6% | 99.5% | 47.1 GB |
| Llama-70B fp16, 8 replicas x 4 A100 | 6s | 17.719s | 32,768 | 44.287s | -149.9% | -85.5% | 343.6 GB |
| Llama-70B fp16, 8 replicas x 4 A100, shallow sweep | 6s | 17.719s | 16,384 | 17.756s | -0.2% | 10.5% | 171.8 GB |
| Llama-70B fp16, 8 replicas x 4 A100 | 40s | 17.719s | 65,536 | 16.816s | 5.1% | 54.0% | 687.2 GB |
| Llama-70B fp16, 8 replicas x 4 A100 | 80s | 17.719s | 65,536 | 16.816s | 5.1% | 54.0% | 687.2 GB |

## Main Read

The result is not GLM-specific. The target mechanism gives large p95 wins on
three Llama-70B/H100 deployments and on the GLM-like H100 deployment.

The result is also not unconditional. On A100 with only 6 seconds of lead time,
32k/full warming is too slow and makes p95 worse. A shallow 16k warm is roughly
p95-neutral and improves mean TTFT by 10.5%. With 40 seconds of lead time, A100
full warming improves mean TTFT by 54.0%, but p95 only by 5.1%.

This suggests a clean gating rule for the algorithm:

- estimate warm time for `(model, hardware, prefix depth)`;
- only warm a depth that can finish before the predicted burst;
- otherwise either warm a shallower prefix or do nothing.

## Paper/Blog Claim Supported By This Matrix

Good claim:

> For synchronized same-prefix bursts, moving prefix prefill off the request
> critical path can reduce p95 TTFT by 67-99% on H100-class simulated
> deployments.

Claim to avoid:

> Predictive warming always beats cache-aware routing.

The A100 rows are the counterexample: the idea needs enough lead/slack relative
to the warm cost.
