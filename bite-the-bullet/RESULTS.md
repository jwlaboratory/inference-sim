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
