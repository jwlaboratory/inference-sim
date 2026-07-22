# BTB Utility-Gate Research

This folder tracks the stronger "bite the bullet" hypothesis:

> Predict whether speculative prefix warming is utility-positive, not merely
> whether same-prefix traffic is coming.

## Harness

`btb_utility_gate.py` tests a workload in three stages:

1. Replay baseline cache-aware/no-remote routing.
2. Create candidate warm triggers from observable prefix/session history.
3. Label each candidate by counterfactual replay: fire only that trigger and
   mark it positive if the selected objective improves.
4. Train a standardized logistic gate on train-window candidates.
5. Select the trigger threshold by train-window replay utility.
6. Evaluate held-out baseline, trigger-all, greedy oracle, and trained gate.

The greedy oracle is intentionally conservative: it considers candidates ranked
by individual counterfactual utility and keeps a trigger only if adding it
improves the combined replay objective.

## Workload Inventory

| Source | Prefix signal | Status | Held-out result |
| --- | --- | --- | --- |
| ART-Chat-2.5M | Real `hash_ids` | Positive | trained gate: mean TTFT -8.5%, p95 TTFT -10.5% |
| Mooncake trace | Real `hash_ids` | Boundary/no-op | no utility-positive triggers in sampled windows |
| BurstGPT v2 | Session-derived blocks | Weak/semi-real | stressed session replay: trained gate p95 TTFT -4.8%, mean ~neutral |

BurstGPT is not Mooncake-compatible as released: v2 has `Timestamp`,
`Session ID`, `Elapsed time`, model, request tokens, response tokens, total
tokens, and log type, but no prompt block hashes. The adapter therefore treats
turns in the same session as sharing deterministic synthetic prefix blocks up
to the observed request-token length. Use BurstGPT results as a session-derived
stress test, not as direct KV-hash evidence.

## Current Results

### ART-Chat-2.5M, Real Hashes

Command:

```bash
python3 experiments/btb_utility_gate.py \
  --dataset alessiotoniolo/ART-Chat-2.5M \
  --config-name '' \
  --block-tokens 256 \
  --model-preset default \
  --num-replicas 4 \
  --gpus-per-replica 4 \
  --rows-per-window 800 \
  --horizon-s 30 \
  --max-candidates-per-window 60 \
  --out experiments/btb_utility_gate_art_h30_replay_threshold_greedy.json
```

Held-out replay:

| Policy | Mean TTFT | p95 TTFT | Triggers/window | Warm GB/window |
| --- | ---: | ---: | ---: | ---: |
| baseline | 43.226s | 86.659s | 0.0 | 0.000 |
| trigger all | 46.736s | 88.223s | 60.0 | 27.962 |
| greedy oracle | 36.281s | 71.602s | 3.0 | 5.369 |
| trained gate | 39.551s | 77.522s | 13.3 | 2.908 |

Takeaway: real non-synthetic win. Triggering every burst-like candidate is
harmful, but the utility-trained gate improves both mean and p95 TTFT.

### Mooncake, Real Hashes

Command:

```bash
python3 experiments/btb_utility_gate.py \
  --rows-per-window 800 \
  --horizon-s 10 \
  --max-candidates-per-window 60 \
  --out experiments/btb_utility_gate_mooncake_h10_greedy.json
```

Held-out replay:

| Policy | Mean TTFT | p95 TTFT | Triggers/window |
| --- | ---: | ---: | ---: |
| baseline | 0.209s | 0.862s | 0.0 |
| trigger all | 0.209s | 0.862s | 60.0 |
| greedy oracle | 0.209s | 0.862s | 0.0 |
| trained gate | 0.209s | 0.862s | 0.0 |

Takeaway: the gate correctly no-ops. This remains a useful negative control.

### BurstGPT v2, Session-Derived Blocks

Command:

```bash
python3 experiments/btb_utility_gate.py \
  --source burstgpt_csv \
  --windows 6 \
  --train-windows 3 \
  --rows-per-window 500 \
  --burstgpt-max-rows 22000 \
  --burstgpt-starts 0 12000 12250 11750 17250 17500 \
  --arrival-scale 0.005 \
  --block-tokens 256 \
  --key-blocks 4 \
  --warm-blocks 4 \
  --model-preset default \
  --num-replicas 4 \
  --gpus-per-replica 4 \
  --horizon-s 6 \
  --active-ttl-s 6 \
  --max-candidates-per-window 60 \
  --include-cold-candidates \
  --out experiments/btb_utility_gate_burstgpt_session_scale0005_h6_greedy.json
```

Held-out replay:

| Policy | Mean TTFT | p95 TTFT | Triggers/window | Warm GB/window |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.076s | 0.205s | 0.0 | 0.000 |
| trigger all | 0.085s | 0.240s | 60.0 | 27.515 |
| greedy oracle | 0.073s | 0.185s | 1.7 | 1.678 |
| trained gate | 0.076s | 0.195s | 14.0 | 12.080 |

Takeaway: weak positive p95 signal under stressed session-derived prefix reuse,
but the trained gate over-warms relative to greedy oracle. This is not yet a
publishable workload win.

## Next Research Loop

- Add a grouped-threshold objective that penalizes warm GB/GPU-seconds, not only
  latency.
- Test BurstGPT with fewer cold-start candidates and larger train windows.
- Search for public agent/eval traces with actual request text or block hashes;
  BurstGPT only gives session IDs.
- Add a small real-hardware replay once an OpenAI-compatible prefix-cache server
  is available.
