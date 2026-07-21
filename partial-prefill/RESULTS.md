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

## Adaptive Idle-Only Prefill

We also tested an adaptive policy:

- On a prediction signal, advance all workers to the signal time.
- Warm only workers with no running or waiting foreground work.
- Warm as much of the prefix as fits before the predicted burst.
- If no worker is idle, skip warming.
- Let normal LRU cache insertion evict stale KV if HBM is full.

In this simulator, a "worker" is a whole tensor-parallel serving node, not an
individual GPU inside the node.

### Default Cache-Aware, `imbalance_abs=8`

Baseline `cache_aware_no_warm`:

- mean TTFT: 6.61s
- p95 TTFT: 10.26s

| Policy/cell | Actual P/R | Delta mean TTFT | Delta p95 TTFT | Warm GB | Warm busy |
|---|---:|---:|---:|---:|---:|
| adaptive_empty_full | 1.00/1.00 | -3.16s | -1.26s | 429.5 | 93.6s |
| fixed_32768 | 1.00/1.00 | -1.98s | -2.02s | 343.6 | 74.9s |
| fixed_65536 | 1.00/1.00 | -2.49s | -0.18s | 687.2 | 149.7s |
| adaptive_empty_full | 0.50/1.00 | -2.43s | +0.91s | 737.3 | 160.6s |
| fixed_32768 | 0.50/1.00 | -1.75s | -0.67s | 687.2 | 149.7s |
| fixed_65536 | 0.50/1.00 | -0.32s | +7.71s | 1374.4 | 299.4s |

Default cache-aware is already close to least-load on this burst, so the p95
bar is high. Fixed 32k warming remains best for p95 on short-output labeling
bursts. Adaptive idle-only improves mean TTFT more, but misses enough warming
opportunities that p95 is worse.

### Stubborn Cache-Aware, `imbalance_abs=512`

Baseline `cache_aware_no_warm`:

- mean TTFT: 10.00s
- p95 TTFT: 14.49s

| Policy/cell | Actual P/R | Delta mean TTFT | Delta p95 TTFT | Warm GB | Warm busy |
|---|---:|---:|---:|---:|---:|
| adaptive_empty_full | 1.00/1.00 | -6.56s | -5.49s | 429.5 | 93.6s |
| fixed_32768 | 1.00/1.00 | -5.38s | -6.25s | 343.6 | 74.9s |
| fixed_65536 | 1.00/1.00 | -5.88s | -4.41s | 687.2 | 149.7s |
| adaptive_empty_full | 0.50/1.00 | -5.83s | -3.32s | 737.3 | 160.6s |
| fixed_32768 | 0.50/1.00 | -5.14s | -4.89s | 687.2 | 149.7s |
| fixed_65536 | 0.50/1.00 | -3.72s | +3.48s | 1374.4 | 299.4s |

When cache-aware waits too long, adaptive idle-only is clearly useful. It does
not beat fixed 32k on p95 in this clean short-output burst, but it does beat
full-prefix warming and uses less background work than full warming.

### Decode-Heavy Variant

We repeated the default cache-aware run with 256 output tokens/request. In this
workload, decode backlog dominates TTFT:

- baseline mean TTFT: 432.53s
- baseline p95 TTFT: 814.11s

| Policy/cell | Actual P/R | Delta mean TTFT | Delta p95 TTFT | Warm GB | Warm busy |
|---|---:|---:|---:|---:|---:|
| adaptive_empty_full | 1.00/1.00 | -4.35s | -0.62s | 85.9 | 18.7s |
| fixed_32768 | 1.00/1.00 | +12.53s | +11.71s | 343.6 | 74.9s |
| fixed_65536 | 1.00/1.00 | +25.52s | +24.96s | 687.2 | 149.7s |
| adaptive_empty_full | 0.50/1.00 | -4.35s | -0.62s | 85.9 | 18.7s |
| fixed_32768 | 0.50/1.00 | +28.61s | +30.01s | 687.2 | 149.7s |
| fixed_65536 | 0.50/1.00 | +60.55s | +73.73s | 1374.4 | 299.4s |

This is the strongest case for adaptive gating. Fixed background warming can
make decode-heavy workloads much worse by adding prefill-decode interference.
The idle-only policy mostly avoids that damage while still getting small wins
from slack.

### Updated Takeaway

The controller should be adaptive, but "idle means warm full" is still too
simple. A practical policy probably needs three gates:

1. **Resource gate:** only warm on idle workers or on workers with enough
   measured prefill slack.
2. **Depth gate:** choose partial depth from confidence, expected burst size,
   and cache pressure.
3. **Objective gate:** optimize mean TTFT, p95 TTFT, or goodput explicitly,
   because the best action changes across short-output labeling bursts and
   decode-heavy generation bursts.

So the current best rule of thumb is:

- short-output labeling burst, high recall: fixed or adaptive 25-50% prefix
  warming is strong;
- noisy predictor: avoid full-prefix warming;
- decode-heavy workload: idle-only adaptive gating is much safer than fixed
  warming.
