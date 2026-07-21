# Bite The Bullet Results

## Mooncake HBM-Only Threshold Sweep

Setup:

- Dataset: `valeriol29/mooncake-traces`, `mooncake` config
- Window sample: 12 windows x 1500 rows
- Prefix key: first 8 Mooncake blocks, 512 tokens/block
- Stress: `--arrival-scale 0.25`
- No remote path: `--rdma-gbps 0 --hbm-only`
- Predictor: `bite-the-bullet/mooncake_burst_model.json`
- Policies:
  - `pure_cache_affinity`: always route to the most local KV worker
  - `cache_aware_no_remote`: cache-aware until imbalance crosses the configured threshold
  - `predict_fake`: predicted background prefill of the shared prefix on extra workers
  - `seed_real`: route a real predicted request cold to a new worker early, so that worker becomes a cache-affine target

| Cache-aware abs | Policy | Mean TTFT | P95 TTFT | Mean latency | Warm GB | Trigger P/R |
|---:|---|---:|---:|---:|---:|---:|
| 8 | pure_cache_affinity | 154.160s | 600.707s | 173.179s | 0.00 | 0.00/0.00 |
| 8 | least_load_no_remote | 61.843s | 127.542s | 82.625s | 0.00 | 0.00/0.00 |
| 8 | cache_aware_no_remote | 59.739s | 139.156s | 80.089s | 0.00 | 0.00/0.00 |
| 8 | predict_fake_b8_t0.80 | 60.237s | 126.553s | 80.545s | 4.03 | 1.00/0.99 |
| 8 | seed_real_b8_t0.36 | 55.436s | 165.105s | 75.722s | 3.47 | 1.00/1.00 |
| 16 | cache_aware_no_remote | 57.856s | 157.120s | 78.196s | 0.00 | 0.00/0.00 |
| 16 | predict_fake_b8_t0.80 | 60.291s | 129.139s | 80.588s | 4.03 | 1.00/0.99 |
| 16 | seed_real_b8_t0.36 | 56.021s | 162.562s | 76.112s | 3.47 | 1.00/1.00 |
| 32 | cache_aware_no_remote | 56.962s | 175.575s | 77.132s | 0.00 | 0.00/0.00 |
| 32 | predict_fake_b8_t0.80 | 60.791s | 128.522s | 81.018s | 4.03 | 1.00/0.99 |
| 32 | seed_real_b8_t0.36 | 55.540s | 168.350s | 75.694s | 3.47 | 1.00/1.00 |
| 64 | cache_aware_no_remote | 57.927s | 188.621s | 78.100s | 0.00 | 0.00/0.00 |
| 64 | predict_fake_b8_t0.80 | 61.972s | 131.584s | 82.228s | 4.03 | 1.00/0.99 |
| 64 | seed_real_b8_t0.36 | 55.796s | 172.818s | 75.827s | 3.47 | 1.00/1.00 |
| 128 | cache_aware_no_remote | 60.373s | 197.041s | 80.509s | 0.00 | 0.00/0.00 |
| 128 | predict_fake_b8_t0.80 | 61.912s | 154.489s | 82.080s | 4.03 | 1.00/0.99 |
| 128 | seed_real_b8_t0.36 | 58.348s | 185.418s | 78.398s | 3.47 | 1.00/1.00 |

## Takeaways

Pure cache affinity confirms the pile-up failure mode: the p95 TTFT reaches
600.7s in this stressed Mooncake slice because requests stay stuck to the hot
KV owner.

Cache-aware routing already performs a reactive real seed. Once its imbalance
guard trips, it sends a real request to a cold worker; that request recomputes
the prefix and leaves KV behind for future cache-affine routing. `seed_real` is
the predictive version of that same action.

The predictor being accurate is necessary but not sufficient. A high-confidence
"more same-prefix requests are coming" prediction only becomes a guaranteed win
when the policy also predicts that cache-aware will need to split the prefix
soon. Otherwise early seeding can improve mean TTFT while hurting p95 by making
some real user request pay the cold prefill before it was unavoidable.

In this sweep, `seed_real` becomes net better than stubborn cache-aware once the
cache-aware fallback waits too long:

- At `abs=32`, `seed_real_b8_t0.36` improves mean TTFT by 1.421s and p95 TTFT
  by 7.224s versus cache-aware.
- At `abs=64`, it improves mean TTFT by 2.132s and p95 TTFT by 15.804s.
- At `abs=128`, it improves mean TTFT by 2.024s and p95 TTFT by 11.622s.

For an aggressive p95 objective, `predict_fake` is often better in this
HBM-only setting because it prepares another cache-affine worker without making
a foreground request pay the cold-prefill cost. Its tradeoff is extra background
compute and HBM footprint.

## Synthetic Labeling-Spike Benchmark

This benchmark creates the trace shape that public traces may underrepresent:
large data-labeling-style fanouts where hundreds of requests reuse the same
long prefix at almost the same time.

Default synthetic setup for the accuracy sweeps:

- 8 true jobs x 500 requests/job
- 65,536 shared prefix tokens/job
- 1 output token/request
- 120 decoy long-prefix jobs for false-positive predictions
- no RDMA and HBM-only cache reuse
- 6s prediction lead time
- 3 predictor-sampling trials per precision/recall cell

### Clean Single-Spike Upper Bound

One 500-request spike, no decoys, no background:

| Policy | Mean TTFT | P95 TTFT | Delta mean | Delta p95 |
|---|---:|---:|---:|---:|
| cache_aware_no_warm | 5.643s | 6.424s | 0.000s | 0.000s |
| pure_cache_affinity | 9.656s | 14.135s | +4.013s | +7.711s |
| predict_fake_prefill, P=1/R=1 | 0.868s | 1.638s | -4.774s | -4.786s |
| predict_seed_real, P=1/R=1 | 6.915s | 8.926s | +1.272s | +2.502s |

This is the best case for fake prefill: the signal arrives early, all warmed
replicas are useful, and there are no false positives. It strongly validates
the idea that public traces can miss the workload where proactive warming is
obviously valuable.

`seed_real` is not the right action for this clean single-spike case. It makes
foreground requests pay cold-prefill cost, while cache-aware already creates
cold real seeds reactively once imbalance appears.

### Accuracy Sweep With Normal Cache-Aware

With `cache_aware_no_warm` spilling at `abs=64`, the baseline already splits
the 500-request spike early:

| Policy/cell | Mean TTFT | P95 TTFT | Delta mean | Delta p95 |
|---|---:|---:|---:|---:|
| cache_aware_no_warm | 6.661s | 10.249s | 0.000s | 0.000s |
| least_load_recompute | 6.601s | 10.258s | -0.059s | +0.009s |
| predict_fake_prefill, P=1/R=1 | 4.120s | 10.079s | -2.540s | -0.170s |
| predict_fake_prefill, P=0.50/R=1 | 6.322s | 17.154s | -0.339s | +6.905s |
| predict_seed_real, P=1/R=1 | 7.326s | 10.648s | +0.665s | +0.399s |

Here fake prefill can improve mean TTFT, but p95 barely improves unless the
false-positive rate is very low. At actual precision around 0.50, fake prefill
still helps mean slightly but loses badly on p95 because wrong long-prefix
predictions burn background prefill work.

### Accuracy Sweep With Stubborn Cache-Aware

With `cache_aware_no_warm` spilling at `abs=512`, cache-aware becomes equivalent
to pure cache affinity for these 500-request bursts:

| Policy/cell | Mean TTFT | P95 TTFT | Delta mean | Delta p95 |
|---|---:|---:|---:|---:|
| cache_aware_no_warm | 10.004s | 14.487s | 0.000s | 0.000s |
| least_load_recompute | 6.601s | 10.258s | -3.403s | -4.229s |
| predict_fake_prefill, P=1/R=1 | 4.120s | 10.079s | -5.884s | -4.408s |
| predict_fake_prefill, P=0.73/R=1 | 4.991s | 13.272s | -5.013s | -1.214s |
| predict_fake_prefill, P=0.50/R=1 | 6.322s | 17.154s | -3.682s | +2.667s |
| predict_fake_prefill, P=0.25/R=1 | 25.213s | 56.731s | +15.209s | +42.244s |
| predict_seed_real, P=1/R=1 | 7.326s | 10.648s | -2.678s | -3.839s |
| predict_seed_real, P=0.25/R=1 | 7.326s | 10.648s | -2.678s | -3.839s |

In this setting, both predictive actions beat stubborn cache-aware when recall
is high. Fake prefill wins harder at high precision, but it becomes dangerous
once precision falls because every false positive costs multiple full-prefix
prefills. `seed_real` has a smaller upside, but it is much less sensitive to
false positives because a wrong signal only has a major cost if a real decoy
request actually arrives while the key is active.

The useful predictor target is therefore not generic "burst probability." It is:

1. Will this prefix receive enough near-future requests to require splitting?
2. Will cache-aware split late enough that early action beats reactive seeding?
3. Is confidence high enough to justify the action cost?

For fake prefill in this synthetic HBM-only benchmark, p95 wins require high
precision. For real-seed, the main requirement is high recall when cache-aware
is too stubborn; precision is less critical, but the improvement is smaller.
