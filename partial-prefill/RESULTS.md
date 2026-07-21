# Partial Prefill Results

## Setup

Synthetic data-labeling spike benchmark:

- 8 true burst jobs x 500 requests/job
- 65,536 shared prefix tokens/job
- 120 decoy long-prefix jobs
- 1 output token/request
- no RDMA, HBM-only
- 6s prediction lead time
- predictor recall fixed at 1.0
- predictor precision swept through approximately 1.0, 0.73, 0.50, 0.25
- 3 predictor-sampling trials per cell

The policy is `partial_fake_prefill`: after a job-level signal, fake-prefill
only the first `N` tokens of the shared prefix on extra workers. Real requests
reuse that partial KV and recompute the rest.

## Normal Cache-Aware, `imbalance_abs=64`

Baseline `cache_aware_no_warm`:

- mean TTFT: 6.66s
- p95 TTFT: 10.25s

Selected cells:

| Warmed tokens | Prefix fraction | Actual P/R | Delta mean TTFT | Delta p95 TTFT | Warm GB | Warm busy |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 3% | 1.00/1.00 | -0.20s | -0.14s | 21.5 | 4.7s |
| 8,192 | 12% | 1.00/1.00 | -0.63s | -0.58s | 85.9 | 18.7s |
| 16,384 | 25% | 1.00/1.00 | -1.18s | -1.08s | 171.8 | 37.4s |
| 32,768 | 50% | 1.00/1.00 | -2.03s | -2.01s | 343.6 | 74.9s |
| 65,536 | 100% | 1.00/1.00 | -2.54s | -0.17s | 687.2 | 149.7s |
| 32,768 | 50% | 0.50/1.00 | -1.96s | -1.37s | 687.2 | 149.7s |
| 65,536 | 100% | 0.50/1.00 | -1.08s | +2.77s | 1374.4 | 299.4s |
| 32,768 | 50% | 0.25/1.00 | +0.05s | +4.51s | 1374.4 | 299.4s |
| 65,536 | 100% | 0.25/1.00 | +11.59s | +31.53s | 2748.8 | 598.8s |

Takeaway: even against a reasonably aggressive cache-aware router, partial
prefill improves p95. Full-prefix prefill gives the best mean TTFT at perfect
precision, but it is fragile. At 50% precision, full prefill hurts p95 while
half-prefix prefill still improves it.

## Stubborn Cache-Aware, `imbalance_abs=512`

Baseline `cache_aware_no_warm`:

- mean TTFT: 10.00s
- p95 TTFT: 14.49s

Selected cells:

| Warmed tokens | Prefix fraction | Actual P/R | Delta mean TTFT | Delta p95 TTFT | Warm GB | Warm busy |
|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 3% | 1.00/1.00 | -3.54s | -4.38s | 21.5 | 4.7s |
| 8,192 | 12% | 1.00/1.00 | -3.97s | -4.81s | 85.9 | 18.7s |
| 16,384 | 25% | 1.00/1.00 | -4.52s | -5.32s | 171.8 | 37.4s |
| 32,768 | 50% | 1.00/1.00 | -5.38s | -6.25s | 343.6 | 74.9s |
| 65,536 | 100% | 1.00/1.00 | -5.88s | -4.41s | 687.2 | 149.7s |
| 32,768 | 50% | 0.50/1.00 | -5.31s | -5.61s | 687.2 | 149.7s |
| 65,536 | 100% | 0.50/1.00 | -4.42s | -1.46s | 1374.4 | 299.4s |
| 32,768 | 50% | 0.25/1.00 | -3.29s | +0.27s | 1374.4 | 299.4s |
| 65,536 | 100% | 0.25/1.00 | +8.24s | +27.29s | 2748.8 | 598.8s |

Takeaway: when cache-aware waits too long, partial prefill is very strong.
The best p95 cell is half-prefix warming, not full-prefix warming. Full-prefix
warming spends twice the background work and becomes much more vulnerable to
false positives.

## Interpretation

This validates the partial-prefill idea. We do not have to warm the entire KV
prefix to create new cache-affine targets. Warming the first 25-50% of a long
shared prompt is enough to:

1. make routing split the burst across workers,
2. reduce each cold worker's prefill cost,
3. cap false-positive damage compared with full-prefix fake prefill.

In this synthetic workload, the practical policy should probably be:

- if predictor confidence is very high and background capacity is idle, consider
  deeper warming;
- if predictor confidence is medium, warm only the first 25-50%;
- if predictor confidence is low, avoid fake-prefill or use a very shallow
  prefix, because full-prefix false positives are brutal.

The next useful extension is an adaptive controller that chooses warm depth from
expected burst size, confidence, current queue load, and available background
compute.
