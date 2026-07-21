# Paper Experiments

## Claim

The valid paper idea is not "KV warming always wins." The defensible claim is:

> Predictive, slack-aware, partial-prefix KV warming creates new cache-affine
> serving targets before a shared-prefix burst arrives. It improves TTFT/goodput
> for metadata-predictable fanout workloads, while adaptive resource and depth
> gates limit harm on noisy or decode-heavy workloads.

The three experiments below are designed to be sufficient for a paper:

1. Show the main win on the target workload.
2. Show why partial/adaptive warming is the right mechanism.
3. Show boundaries and robustness on non-target workloads and stronger systems.

Each experiment should be run on the same real serving stack where possible
and mirrored in the simulator for fast ablation.

## Common Real-System Setup

Primary target:

- Cluster: 8 serving replicas.
- Replica shape: 1 node x 8 H100 80GB SXM GPUs, tensor parallelism `TP=8`.
- Total GPUs: 64 H100.
- Intra-node: NVLink/NVSwitch.
- Inter-node: 400Gb/s or 800Gb/s InfiniBand.
- Serving engine: vLLM or SGLang with prefix caching enabled. A Dynamo/SGLang
  gateway can be used for KV/cache-aware routing baselines.
- Model: GLM-5.2-like 744B MoE, quantized weights.
  - Total parameters: 744B.
  - Active parameters/token: 40B.
  - Layers: 78.
  - KV geometry: MLA-like compressed KV, modeled as `KV_HEADS=1`,
    `HEAD_DIM=288`.
  - Weight dtype: int4 or fp8. bf16 744B does not fit on 8xH100
    (`744B * 2B = 1.49TB` weights vs `8 * 80GB = 640GB` HBM).
  - KV dtype: fp8 or fp16, recorded explicitly.
- Context limit: at least 128k tokens. If using real GLM-5.2 with larger
  context, cap benchmark prompts to 128k unless the cluster can sustain 1M.
- Prefix cache block size: 512 tokens for Mooncake-like traces; 256 or 512 is
  acceptable for synthetic workloads if reported.
- Router defaults:
  - Cache-aware: route to highest local prefix match.
  - Default spill threshold: `balance_abs=8`, `balance_rel=1.5`.
  - Stubborn spill threshold for stress: `balance_abs=512`, `balance_rel=1.5`.
- Decode scheduling: continuous batching, max batch 256 sequences/replica.

Simulator equivalent:

- `PARAMS=744e9`
- `ACTIVE_PARAMS=40e9`
- `LAYERS=78`
- `KV_HEADS=1`
- `HEAD_DIM=288`
- `CLUSTER=8 replicas x 8 H100`
- Run both `DTYPE_BYTES=0.5` for int4-like weight/KV stress and `DTYPE_BYTES=1`
  for fp8-like weight/KV stress. If the simulator later decouples weight dtype
  from KV dtype, use int4/fp8 weights and fp8/fp16 KV separately.

## Experiment 1: Target Workload Win

### Thesis

For metadata-predictable data-labeling or agent-fanout jobs, proactive partial
warming reduces TTFT versus default cache-aware routing because the router has
cache-affine targets ready before the burst queues.

### Setup

- Hardware/model: common real-system setup above.
- Network condition: HBM-only prefix reuse for the primary result.
  - Disable remote KV transfer/RDMA fetch for this experiment, or report it as
    a separate row.
- Workload: synthetic but product-realistic data-labeling fanout.
  - Jobs: 20 independent jobs.
  - Shared prefix/job: 64k and 128k token variants.
  - Requests/job: 500.
  - Arrival shape: all 500 arrive over a 1s window.
  - Unique suffix/request: 128 or 256 tokens.
  - Decode length/request: 1 token and 8 token variants.
  - Job spacing: at least 30s between jobs for the clean main figure.
  - Background: 5-10% unrelated requests with 2k-8k prompts and 32 output
    tokens.
- Predictor signal:
  - Signal arrives 6s before burst start.
  - Precision/recall for the headline run: 1.0/1.0.
  - Signal source in real stack: job metadata, API batch id, tenant/job id, or
    agent phase id.

### Policies

- `cache_aware_default`: cache-aware with `abs=8`, `rel=1.5`.
- `least_load`: shortest queue/load, no cache preference.
- `pure_cache_affinity`: always route to the current prefix owner.
- `full_fake_prefill`: warm 100% of the shared prefix on extra replicas.
- `partial_prefill_25`: warm first 25% of the shared prefix.
- `partial_prefill_50`: warm first 50% of the shared prefix.
- `adaptive_partial`: warm up to 50% only on idle replicas; skip if none idle.

### Metrics

- TTFT: mean, p50, p95, p99.
- End-to-end latency: mean, p95.
- Goodput: requests meeting a TTFT SLO, e.g. 2s/5s/10s depending model.
- Warm cost: GPU-seconds spent warming, HBM GB inserted, number of warmed
  replicas.
- Queue depth per replica over time.
- Cache hit ratio by prefix segment.

### Expected Figure

One bar chart:

- x-axis: policy.
- y-axis: p95 TTFT.
- bars grouped by 64k and 128k prefix.
- annotate warm GPU-seconds above each predictive policy.

### Success Criteria

The paper claim is supported if:

- `partial_prefill_25` or `partial_prefill_50` improves p95 TTFT by at least
  20% versus `cache_aware_default`.
- It uses at most 60% of the warm compute of `full_fake_prefill`.
- `full_fake_prefill` improves mean TTFT but is worse than partial prefill on
  p95 or warm cost.

### Simulator Command

Current smaller-model proxy:

```bash
python3 partial-prefill/sweep_partial_prefill.py \
  --imbalance-abs 8 \
  --num-bursts 8 \
  --burst-size 500 \
  --prefix-tokens 65536 \
  --output-tokens 1 \
  --precision-sweep 1 \
  --recall 1 \
  --warm-tokens 16384 32768 65536 \
  --out partial-prefill/paper_exp1_target_win.json
```

## Experiment 2: Predictor Accuracy And Warm-Depth Frontier

### Thesis

Partial warming is the robust policy because it lies on the Pareto frontier
between benefit and false-positive cost. Full-prefix warming has higher upside
only when predictions are very accurate and background slack is plentiful.

### Setup

- Hardware/model: common real-system setup above.
- Network condition: HBM-only primary; RDMA disabled to isolate recompute
  versus proactive prefill.
- Workload:
  - Same as Experiment 1, 64k prefix only.
  - Jobs: 20 true burst jobs.
  - Decoys: 200 false-positive candidate jobs.
  - Decoy request count: 1 request/job for metadata false positives.
  - Decode length: 1 token.
- Predictor sweep:
  - Recall: 1.0 for the main precision frontier.
  - Precision: 1.0, 0.9, 0.75, 0.5, 0.25.
  - Secondary sweep: recall 0.25, 0.5, 0.75, 1.0 at precision 0.75.
- Warm depth sweep:
  - 0, 2k, 4k, 8k, 16k, 32k, 64k tokens.
  - Adaptive confidence-scaled depth: `warm_depth = max_depth * confidence`,
    rounded to a block boundary.

### Policies

- `cache_aware_default`.
- Fixed warm depths: `partial_2k`, `partial_4k`, `partial_8k`, `partial_16k`,
  `partial_32k`, `full_64k`.
- `adaptive_idle_partial`: idle-only, up to 50%.
- `adaptive_confidence_depth`: idle-only, depth scaled by confidence.

### Metrics

- p95 TTFT delta versus `cache_aware_default`.
- Mean TTFT delta.
- Warm GPU-seconds.
- Warm HBM GB.
- False-positive harm: p95 delta on decoy/background requests.
- Pareto frontier: p95 TTFT improvement versus warm GPU-seconds.

### Expected Figure

Two panels:

1. Heatmap: precision x warm depth, color = p95 TTFT delta.
2. Pareto plot: warm GPU-seconds versus p95 TTFT delta, with partial depths
   labeled.

### Success Criteria

The paper claim is supported if:

- 25-50% prefix warming is on or near the Pareto frontier across precision
  levels.
- At precision 0.5, 25-50% warming still beats or matches cache-aware p95,
  while full warming regresses.
- Adaptive depth reduces false-positive harm versus full warming.

### Simulator Command

```bash
python3 partial-prefill/sweep_partial_prefill.py \
  --imbalance-abs 8 \
  --num-bursts 8 \
  --burst-size 500 \
  --prefix-tokens 65536 \
  --output-tokens 1 \
  --precision-sweep 1 0.75 0.5 0.25 \
  --recall 1 \
  --trials 5 \
  --warm-tokens 2048 4096 8192 16384 32768 65536 \
  --out partial-prefill/paper_exp2_depth_frontier.json
```

Adaptive variant:

```bash
python3 partial-prefill/adaptive_partial_prefill.py \
  --imbalance-abs 8 \
  --num-bursts 8 \
  --burst-size 500 \
  --prefix-tokens 65536 \
  --output-tokens 1 \
  --precision-sweep 1 0.75 0.5 0.25 \
  --trials 5 \
  --fixed-warm-tokens 32768 65536 \
  --out partial-prefill/paper_exp2_adaptive.json
```

## Experiment 3: Robustness And Boundary Conditions

### Thesis

The mechanism should help target burst workloads without pretending to dominate
all LLM serving. It should degrade gracefully on trace replay, decode-heavy
generation, and RDMA-enabled systems.

### Setup

- Hardware/model: common real-system setup above.
- Workloads:
  1. Public trace replay:
     - Mooncake `mooncake` config or internal production trace with timestamp,
       input length, output length, and prefix block ids.
     - Replay 12 windows x 1500 requests/window.
     - Preserve recorded input/output lengths.
  2. Decode-heavy fanout:
     - Same fanout as Experiment 1.
     - Decode length/request: 256 tokens.
  3. Mixed online workload:
     - Same fanout as Experiment 1.
     - Add 500-2000 background requests over the same interval.
     - Background decode length: 64-256 tokens.
- Network matrix:
  - HBM-only/no remote KV.
  - RDMA fetch enabled at 50GB/s/GPU.
  - RDMA fetch enabled at 100GB/s/GPU, if hardware supports it.
- Cache-aware settings:
  - Default: `abs=8`, `rel=1.5`.
  - Stubborn: `abs=512`, `rel=1.5`.

### Policies

- `cache_aware_default`.
- `least_load`.
- `reactive_rdma_copy` when RDMA is enabled.
- `fixed_32k_partial`.
- `adaptive_idle_partial`.
- `adaptive_idle_partial_with_noop`: identical controller, but does nothing if
  predicted warm cost would overlap foreground decode.

### Metrics

- TTFT mean/p95/p99.
- End-to-end latency mean/p95.
- Decode throughput and goodput.
- Warm GPU-seconds.
- Foreground interference: increase in p95 latency for non-burst/background
  requests.
- Win/loss table by workload and network.

### Expected Figure

One table:

| Workload | Network | Best predictive policy | p95 TTFT delta | Warm cost | Background harm |

Plus one line plot for decode-heavy:

- x-axis: output length.
- y-axis: p95 TTFT delta.
- compare fixed partial versus adaptive idle-only.

### Success Criteria

The paper claim is supported if:

- On public trace replay, adaptive partial warming is near-neutral when no true
  large burst exists.
- On decode-heavy workloads, adaptive idle-only is safer than fixed warming and
  avoids large p95 regressions.
- With RDMA enabled, gains shrink or disappear versus reactive RDMA copy; this
  defines the boundary condition rather than invalidating the idea.
- On HBM-only/no-RDMA target bursts, predictive partial warming remains the
  best or near-best policy.

### Simulator Commands

Trace replay:

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
  --imbalance-abs 8 \
  --out bite-the-bullet/paper_exp3_trace_abs8.json
```

Decode-heavy fanout:

```bash
python3 partial-prefill/adaptive_partial_prefill.py \
  --imbalance-abs 8 \
  --output-tokens 256 \
  --background-output-tokens 128 \
  --precision-sweep 1 0.75 0.5 0.25 \
  --trials 5 \
  --fixed-warm-tokens 32768 65536 \
  --out partial-prefill/paper_exp3_decode_heavy.json
```

RDMA boundary:

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
  --out bite-the-bullet/paper_exp3_rdma_boundary.json
```

## Paper Storyboard

The three experiments support this narrative:

1. **Opportunity:** Large fanout jobs with long shared prefixes create a
   cache-affinity/load-balance conflict that default routing cannot fully solve.
2. **Mechanism:** Warming only part of the prefix is the robust point on the
   benefit/cost frontier.
3. **Controller:** The policy must be adaptive to workload and resources; it
   should use idle slack and avoid decode-heavy interference.

If all three pass, the paper has a valid core contribution. If Experiment 1
passes but Experiment 3 fails, the idea is still useful but should be framed as
a specialized batch/agent serving optimization rather than a general LLM serving
router.

