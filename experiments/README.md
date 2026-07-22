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
5. Select the trigger threshold, and optionally a per-window trigger budget,
   by train-window replay utility.
6. Evaluate held-out baseline, trigger-all, greedy oracle, and trained gate.

The greedy oracle is intentionally conservative: it considers candidates ranked
by individual counterfactual utility and keeps a trigger only if adding it
improves the combined replay objective.

## Workload Inventory

| Source | Prefix signal | Status | Held-out result |
| --- | --- | --- | --- |
| ART-Chat-2.5M | Real `hash_ids` | Split-sensitive positive | H30/K8 seed23: mean TTFT -8.5%, p95 TTFT -10.5%; robustness sweep is mixed |
| Mooncake trace | Real `hash_ids` | Boundary/no-op | no utility-positive triggers in sampled windows |
| Qwen Bailian To-C | Real `hash_ids` | Weak positive | random windows: trained gate mean TTFT -0.9%, p95 ~neutral |
| Qwen Bailian To-B | Real `hash_ids` | Boundary/no-op | permissive key: trained gate no-ops on held-out |
| Qwen Bailian Thinking | Real `hash_ids` | Split-sensitive | 3 random warm-cost splits: mean TTFT -1.4% +/- 2.4%, but composite cost worsens |
| Qwen Bailian Coder | Real `hash_ids` | Cost-aware no-op | warm-cost gate no-ops; mean-only gate was harmful |
| BurstGPT v2 | Session-derived blocks | Weak/semi-real | stressed session replay: trained gate p95 TTFT -4.8%, mean ~neutral |

BurstGPT is not Mooncake-compatible as released: v2 has `Timestamp`,
`Session ID`, `Elapsed time`, model, request tokens, response tokens, total
tokens, and log type, but no prompt block hashes. The adapter therefore treats
turns in the same session as sharing deterministic synthetic prefix blocks up
to the observed request-token length. Use BurstGPT results as a session-derived
stress test, not as direct KV-hash evidence.

The harness also supports a generic `--source jsonl` adapter for future
workloads. By default it expects fields named `timestamp`, `input_length`,
`output_length`, and `hash_ids`; all field names are configurable. If `hash_ids`
is absent, the adapter can derive synthetic session-prefix blocks from
`session_id`, which is useful for exploratory session workloads but should be
reported as semi-real rather than true KV-hash evidence.

The utility objective can charge warm work in addition to latency:

```text
objective = metric + warm_gb_cost * warm_gb
                 + warm_busy_cost * warm_busy_s
                 + trigger_cost * triggers
```

This keeps the old latency-only behavior by default while allowing the gate to
learn "worth it after warming overhead" rather than "any latency drop at any
cost."

For more conservative calibration, `--threshold-windows N` reserves the last
`N` pre-test windows for threshold/top-k selection instead of selecting on the
same windows used to fit the logistic model. This is useful for detecting score
calibration drift across trace regions.

Example generic JSONL command:

```bash
python3 experiments/btb_utility_gate.py \
  --source jsonl \
  --jsonl-path path/to/workload.jsonl \
  --block-tokens 256 \
  --key-blocks 4 \
  --warm-blocks 4
```

Qwen Bailian is the best next public workload family found so far. It ships
production-derived JSONL traces with salted `hash_ids` at 16 tokens/block. To
reproduce the local runs below:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon.git \
  /tmp/qwen-bailian-usagetraces-anon
git -C /tmp/qwen-bailian-usagetraces-anon lfs pull \
  --include='qwen_traceA_blksz_16.jsonl,qwen_traceB_blksz_16.jsonl,qwen_thinking_blksz_16.jsonl,qwen_coder_blksz_16.jsonl'
```

Summarize result artifacts:

```bash
python3 experiments/btb_result_summary.py experiments/btb_utility_gate_*.json
```

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

Takeaway: real-hash workload win on this setting. Triggering every burst-like
candidate is harmful, but the utility-trained gate improves both mean and p95
TTFT.

ART robustness sweep:

```bash
python3 experiments/btb_result_summary.py \
  experiments/btb_utility_gate_art_h30_replay_threshold_greedy.json \
  experiments/btb_utility_gate_art_h10_k8.json \
  experiments/btb_utility_gate_art_h30_k4.json \
  experiments/btb_utility_gate_art_h30_k8_seed37.json \
  experiments/btb_utility_gate_art_h60_k8_r1600.json
```

| Variant | Seed | Rows/window | Horizon | Key/warm blocks | Trained dMean | Trained dP95 | Greedy dMean | Greedy dP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H30/K8 | 23 | 800 | 30s | 8/8 | -8.5% | -10.5% | -16.1% | -17.4% |
| H10/K8 | 23 | 800 | 10s | 8/8 | +6.8% | +8.7% | -1.4% | +2.3% |
| H30/K4 | 23 | 800 | 30s | 4/4 | -6.6% | -4.6% | -16.1% | -18.6% |
| H30/K8 | 37 | 800 | 30s | 8/8 | +14.5% | +0.7% | -21.0% | +5.3% |
| H60/K8 | 23 | 1600 | 60s | 8/8 | +2.1% | +4.4% | -1.7% | +7.8% |

Robustness takeaway: ART is promising but not yet a general setting win. The
positive result survives a smaller prefix key at the same 30s horizon, but it
does not survive the 10s horizon, 60s horizon, or one alternate random split.
The seed37 split is especially informative because greedy still finds mean
headroom while the learned gate hurts held-out mean TTFT, which points to score
calibration/distribution shift rather than absence of opportunity.

A one-window validation threshold/top-k selection avoids the bad seed37 trigger
choices, but only by selecting no-op. It also removes the good seed23 H30/K8
speedup:

| Variant | Seed | Threshold windows | Trained dMean | Trained dP95 | Triggers/window |
| --- | ---: | ---: | ---: | ---: | ---: |
| H30/K8 | 23 | 1 | +0.0% | +0.0% | 0.0 |
| H30/K8 | 37 | 1 | +0.0% | +0.0% | 0.0 |

Using more ART windows for the bad seed37 split gives the classifier more
trace coverage and recovers some tail benefit, but still not a mean TTFT win:

| Variant | Seed | Windows | Train windows | Threshold windows | Trained dMean | Trained dP95 | Greedy dMean | Greedy dP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H30/K8 | 37 | 10 | 6 | 0 | +1.9% | -4.9% | -12.5% | -15.1% |
| H30/K8 | 37 | 10 | 6 | 2 | +5.5% | -5.9% | -12.5% | -15.1% |

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

### Qwen Bailian To-C, Real Hashes

Command:

```bash
python3 experiments/btb_utility_gate.py \
  --source jsonl \
  --jsonl-path /tmp/qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl \
  --windows 8 \
  --train-windows 4 \
  --rows-per-window 600 \
  --arrival-scale 1 \
  --block-tokens 16 \
  --key-blocks 32 \
  --warm-blocks 32 \
  --model-preset default \
  --num-replicas 4 \
  --gpus-per-replica 4 \
  --horizon-s 30 \
  --future-k 2 \
  --active-ttl-s 30 \
  --max-candidates-per-window 60 \
  --include-cold-candidates \
  --out experiments/btb_utility_gate_qwen_traceA_blksz_16_random_h30_k32.json
```

Held-out replay:

| Policy | Mean TTFT | p95 TTFT | Triggers/window | Warm GB/window |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.203s | 0.748s | 0.0 | 0.000 |
| trigger all | 0.200s | 0.758s | 60.0 | 0.084 |
| greedy oracle | 0.190s | 0.725s | 2.2 | 0.084 |
| trained gate | 0.201s | 0.748s | 0.5 | 0.000 |

Takeaway: weak real-hash positive. The learned gate avoids most waste and gets a
small mean TTFT win, but it leaves most oracle headroom unused.

### Qwen Bailian To-B, Real Hashes

Command:

```bash
python3 experiments/btb_utility_gate.py \
  --source jsonl \
  --jsonl-path /tmp/qwen-bailian-usagetraces-anon/qwen_traceB_blksz_16.jsonl \
  --windows 6 \
  --train-windows 3 \
  --rows-per-window 600 \
  --jsonl-sequential \
  --arrival-scale 1 \
  --block-tokens 16 \
  --key-blocks 8 \
  --warm-blocks 32 \
  --model-preset default \
  --num-replicas 4 \
  --gpus-per-replica 4 \
  --horizon-s 10 \
  --future-k 1 \
  --active-ttl-s 10 \
  --max-candidates-per-window 60 \
  --include-cold-candidates \
  --out experiments/btb_utility_gate_qwen_traceB_blksz_16_h10_k8.json
```

Held-out replay:

| Policy | Mean TTFT | p95 TTFT | Triggers/window |
| --- | ---: | ---: | ---: |
| baseline | 0.057s | 0.188s | 0.0 |
| trigger all | 0.057s | 0.188s | 60.0 |
| greedy oracle | 0.057s | 0.188s | 0.0 |
| trained gate | 0.057s | 0.188s | 0.0 |

Takeaway: useful negative control. The API/task trace is mostly single-turn; the
trained gate no-ops on held-out.

### Qwen Bailian Thinking, Warm-Cost Real Hashes

Command:

```bash
python3 experiments/btb_utility_gate.py \
  --source jsonl \
  --jsonl-path /tmp/qwen-bailian-usagetraces-anon/qwen_thinking_blksz_16.jsonl \
  --windows 8 \
  --train-windows 4 \
  --rows-per-window 600 \
  --arrival-scale 1 \
  --block-tokens 16 \
  --key-blocks 32 \
  --warm-blocks 32 \
  --model-preset default \
  --num-replicas 4 \
  --gpus-per-replica 4 \
  --horizon-s 60 \
  --future-k 2 \
  --active-ttl-s 60 \
  --max-candidates-per-window 60 \
  --include-cold-candidates \
  --warm-gb-cost 1.0 \
  --gate-topk-options 1 2 4 8 0 \
  --threshold-grid-step 0.1 \
  --no-threshold-score-candidates \
  --out experiments/btb_utility_gate_qwen_thinking_blksz_16_random_h60_k32_warmcost.json
```

Held-out replay:

| Policy | Mean TTFT | p95 TTFT | Triggers/window | Warm GB/window |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.364s | 1.496s | 0.0 | 0.000 |
| trigger all | 0.533s | 1.490s | 60.0 | 0.503 |
| greedy oracle | 0.349s | 1.428s | 0.8 | 0.000 |
| trained gate | 0.349s | 1.443s | 14.8 | 0.084 |

Takeaway: cost-aware utility fixes the previous mean-only failure on this
split. The old latency-only gate over-triggered and worsened held-out mean TTFT
by 8.3%; charging warm GB reduces warm work and improves both mean and p95 TTFT.
However, this does not yet generalize strongly across random splits; see the
split-stability pilot below.

### Qwen Bailian Thinking, Split-Stability Pilot

Additional random-split commands; the seed-23 artifact is the single-split
result above:

```bash
for seed in 11 37; do
  python3 experiments/btb_utility_gate.py \
    --source jsonl \
    --jsonl-path /tmp/qwen-bailian-usagetraces-anon/qwen_thinking_blksz_16.jsonl \
    --windows 8 \
    --train-windows 4 \
    --rows-per-window 600 \
    --seed "$seed" \
    --arrival-scale 1 \
    --block-tokens 16 \
    --key-blocks 32 \
    --warm-blocks 32 \
    --model-preset default \
    --num-replicas 4 \
    --gpus-per-replica 4 \
    --horizon-s 60 \
    --future-k 2 \
    --active-ttl-s 60 \
    --max-candidates-per-window 60 \
    --include-cold-candidates \
    --warm-gb-cost 1.0 \
    --gate-topk-options 1 2 4 8 0 \
    --threshold-grid-step 0.1 \
    --no-threshold-score-candidates \
    --out experiments/btb_utility_gate_qwen_thinking_blksz_16_random_h60_k32_warmcost_seed${seed}.json
done
```

Grouped result:

```bash
python3 experiments/btb_result_summary.py --group-by-label \
  experiments/btb_utility_gate_qwen_thinking_blksz_16_random_h60_k32_warmcost.json \
  experiments/btb_utility_gate_qwen_thinking_blksz_16_random_h60_k32_warmcost_seed11.json \
  experiments/btb_utility_gate_qwen_thinking_blksz_16_random_h60_k32_warmcost_seed37.json
```

| Workload | N | dObjective | Trained dMean | Trained dP95 | Trig/win | Warm GB/win | Greedy dMean | Greedy dP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen_thinking_blksz_16, warm-cost | 3 | +0.1767 +/- 0.2105 | -1.4% +/- 2.4% | -0.7% +/- 2.6% | 6.4 +/- 7.4 | 0.182 +/- 0.207 | -2.5% +/- 2.2% | -2.5% +/- 2.3% |

Validation-threshold variant:

```bash
for seed in 23 11 37; do
  python3 experiments/btb_utility_gate.py \
    --source jsonl \
    --jsonl-path /tmp/qwen-bailian-usagetraces-anon/qwen_thinking_blksz_16.jsonl \
    --windows 8 \
    --train-windows 4 \
    --threshold-windows 2 \
    --rows-per-window 600 \
    --seed "$seed" \
    --arrival-scale 1 \
    --block-tokens 16 \
    --key-blocks 32 \
    --warm-blocks 32 \
    --model-preset default \
    --num-replicas 4 \
    --gpus-per-replica 4 \
    --horizon-s 60 \
    --future-k 2 \
    --active-ttl-s 60 \
    --max-candidates-per-window 60 \
    --include-cold-candidates \
    --warm-gb-cost 1.0 \
    --gate-topk-options 1 2 4 8 0 \
    --threshold-grid-step 0.1 \
    --no-threshold-score-candidates \
    --out experiments/btb_utility_gate_qwen_thinking_blksz_16_random_h60_k32_warmcost_valid_seed${seed}.json
done
```

```bash
python3 experiments/btb_result_summary.py --group-by-label \
  experiments/btb_utility_gate_qwen_thinking_blksz_16_random_h60_k32_warmcost_valid_seed23.json \
  experiments/btb_utility_gate_qwen_thinking_blksz_16_random_h60_k32_warmcost_valid_seed11.json \
  experiments/btb_utility_gate_qwen_thinking_blksz_16_random_h60_k32_warmcost_valid_seed37.json
```

| Workload | N | dObjective | Trained dMean | Trained dP95 | Trig/win | Warm GB/win | Greedy dMean | Greedy dP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen_thinking_blksz_16, warm-cost + validation threshold | 3 | +0.5533 +/- 0.3137 | +2.2% +/- 3.8% | +0.5% +/- 1.0% | 12.6 +/- 10.8 | 0.545 +/- 0.328 | -2.5% +/- 2.2% | -2.5% +/- 2.3% |

Takeaway: Qwen Thinking has real oracle headroom, but the current logistic gate
is not robust enough to call this a publishable win. A validation-window
threshold split surfaces score calibration drift rather than fixing it.

### Qwen Bailian Coder, Warm-Cost Real Hashes

Command:

```bash
python3 experiments/btb_utility_gate.py \
  --source jsonl \
  --jsonl-path /tmp/qwen-bailian-usagetraces-anon/qwen_coder_blksz_16.jsonl \
  --windows 6 \
  --train-windows 3 \
  --rows-per-window 600 \
  --jsonl-sequential \
  --arrival-scale 1 \
  --block-tokens 16 \
  --key-blocks 32 \
  --warm-blocks 32 \
  --model-preset default \
  --num-replicas 4 \
  --gpus-per-replica 4 \
  --horizon-s 60 \
  --future-k 2 \
  --active-ttl-s 60 \
  --max-candidates-per-window 60 \
  --include-cold-candidates \
  --warm-gb-cost 1.0 \
  --gate-topk-options 1 2 4 8 0 \
  --threshold-grid-step 0.1 \
  --no-threshold-score-candidates \
  --out experiments/btb_utility_gate_qwen_coder_blksz_16_h60_k32_cost.json
```

Held-out replay:

| Policy | Mean TTFT | p95 TTFT | Triggers/window | Warm GB/window |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.304s | 0.928s | 0.0 | 0.000 |
| trigger all | 0.324s | 0.936s | 60.0 | 3.020 |
| greedy oracle | 0.304s | 0.928s | 0.0 | 0.000 |
| trained gate | 0.304s | 0.928s | 0.0 | 0.000 |

Takeaway: with warm work charged, coder becomes a no-op. This is better than the
latency-only trained gate, which warmed 1.174 GB/window and worsened held-out
mean TTFT by 4.3%.

## Next Research Loop

- Fix ART score calibration before claiming generality: the first
  validation-window check avoids harm by selecting no-op, so the next version
  needs confidence-aware trigger selection rather than a single replay
  threshold.
- Add tail-aware objective calibration and rerun p95-targeted ART/Qwen sweeps.
- Tune warm-cost units against real serving measurements instead of treating
  `warm_gb_cost=1.0` as final.
- Add regime features for distribution shift: per-window load, candidate score
  calibration, recent queueing, and candidate density.
- Run K-fold/random-window evaluation over ART and Qwen before calling any new
  result publishable.
- Add a small real-hardware replay once an OpenAI-compatible prefix-cache server
  is available.
