# We Built An Open-Source Simulator For LLM Inference Scheduling

Status: draft.

Working title options:

- We Built An Open-Source Simulator For LLM Inference Scheduling
- A Small Simulator For Reasoning About LLM Inference Queues
- Before Renting 64 H100s, Simulate The Queue
- Inference-Sim: A Playground For Prefix Caches, Routers, And GPU Queues

## Narrative Plan

1. Open with the problem: inference systems are hard to reason about because
   small routing choices interact with prefix caches, queues, batching, and GPU
   memory.
2. Explain why dashboards alone are not enough: production traces tell you what
   happened, but not always what would have happened under a different router.
3. Introduce `inference-sim`: a small, composable simulator for replaying LLM
   inference traces across configurable GPU clusters.
4. Explain what it models:
   continuous batching, prefill, decode, prefix-cache tiers, RDMA, disk, host
   RAM, GPU specs, tensor-parallel replicas, and pluggable routing policies.
5. Show why this matters with one concrete case study: same-prefix burst
   scheduling and predictive KV warming.
6. Be honest about the simulator boundary: mechanism validation, not a
   substitute for hardware benchmarking.
7. Invite people to use it to test routers, traces, hardware choices, and cache
   policies.
8. End with launch roadmap: docs, examples, more traces, plugin policies, and
   better dtype modeling.

## Draft

LLM inference systems are full of tradeoffs that sound simple until you try to
reason about them precisely.

Should a request go to the GPU that already has its prefix cached, or to the GPU
with the shortest queue? Should the scheduler prefill aggressively, or protect
decode throughput? Is a larger batch helping throughput, or is it just hiding a
tail-latency problem? If a prefix cache lives on another node, is it faster to
fetch over RDMA or recompute?

The annoying answer is usually: it depends.

It depends on the model shape, the KV layout, the GPU, the interconnect, the
prompt length, the output length, the batch size, and the exact arrival pattern.
The same routing policy can look brilliant on one workload and terrible on
another.

That is why we built `inference-sim`: a small open-source simulator for LLM
inference clusters.

The goal is not to replace real benchmarking. The goal is to make inference
scheduling ideas cheap to test before you turn them into a distributed systems
project.

## The Problem: Routing Is Not A Local Decision

In a serving cluster, a request does not just consume "one GPU slot."

It moves through several coupled systems:

- a queue;
- a prefill stage that reads the input prompt;
- a decode stage that emits tokens one at a time;
- a prefix cache that may or may not already contain useful KV;
- a batching system that groups active decodes;
- a memory budget shared by model weights and KV cache;
- a network or storage path if cache is remote.

This coupling is why intuitive routing policies can fail.

Cache affinity is a good example. If a request shares a long prefix with earlier
requests, routing it to the same replica can save a large prefill. But if 500
requests with that prefix arrive in one second, strict cache affinity may build
a massive queue on one replica while other replicas are idle.

Least-load routing has the opposite failure mode. It spreads the queue, but may
force multiple replicas to recompute the same long prefix.

Cache-aware routing tries to balance both, but the question remains: under which
workloads and hardware configurations does it actually win?

That is the kind of question a simulator should make easy to ask.

## What `inference-sim` Models

`inference-sim` is intentionally small. The core files are:

- `config.py`: every tunable in one place;
- `workload.py`: trace loading and request construction;
- `gpu.py`: node timing equations and prefix-cache behavior;
- `router.py`: routing policies;
- `simulate.py`: event loop;
- `server.py`: API for the UI;
- `ui/`: a React UI for editing configs and visualizing runs.

The simulator replays a window from an inference trace and routes each request
through a configurable cluster.

It models:

- continuous batching;
- prefill timing;
- decode timing;
- model weight memory;
- KV cache memory;
- prefix block reuse;
- HBM, host RAM, disk, and peer-node cache tiers;
- RDMA bandwidth;
- configurable GPU specs;
- tensor-parallel replicas;
- pluggable routing policies.

The timing model is deliberately transparent.

Prefill is modeled as compute-bound:

```text
prefill_time = 2 * active_params * tokens / (flops * MFU)
```

Decode is modeled as memory-bound across the whole active batch:

```text
decode_step_time =
  (active_weight_bytes + batch_kv_bytes) / (hbm_bandwidth * MBU)
```

Cache movement is modeled as:

```text
cache_load_time = kv_bytes / tier_bandwidth
```

These equations are not a full production serving stack. They are a set of
simple, inspectable assumptions that make routing experiments reproducible.

## Why Prefix Blocks Matter

The simulator works with block-level prompt hashes. Two requests share prefix
cache if their leading block hashes match.

That lets us ask questions like:

- How much TTFT comes from queueing versus prefill?
- Does cache-aware routing improve or worsen p95 latency?
- When is it faster to fetch remote KV than recompute?
- How much HBM is consumed by a warmed prefix?
- What happens if a batch of similar requests arrives at once?

The last question turned into one of our first case studies.

## Case Study: Same-Prefix Bursts

Consider this workload:

- 8 burst jobs;
- 500 requests per burst;
- 65,536-token shared prefix;
- 256-token unique suffix per request;
- 1 output token;
- burst window: 1 second;
- max serving batch: 256;
- no RDMA;
- HBM-only cache.

This models labeling, scoring, extraction, and agent fanout workloads where many
requests share a long context but ask for tiny outputs.

On a GLM-like 744B MoE proxy with 8 replicas x 8 H100s, cache-aware routing got
p95 TTFT to 1.313 seconds. Then we tested speculative prefix warming: prefill
part of the shared prefix on extra replicas before the burst arrives.

| Warm depth | p95 TTFT | p95 reduction vs cache-aware | Warm KV | Warm busy time |
| --- | ---: | ---: | ---: | ---: |
| 16,384 tokens | 0.982s | 25.2% | 11.8 GB | 10.6s |
| 32,768 tokens | 0.651s | 50.4% | 23.6 GB | 21.2s |
| 65,536 tokens | 0.005s | 99.6% | 47.1 GB | 42.4s |

We also ran the same burst on standard Llama-70B/H100 configurations:

| System | Lead time | Cache-aware p95 | Best warm p95 | p95 reduction |
| --- | ---: | ---: | ---: | ---: |
| Llama-70B fp16, 4 replicas x 4 H100 | 6s | 6.333s | 1.638s | 74.1% |
| Llama-70B fp16, 8 replicas x 4 H100 | 6s | 5.047s | 1.638s | 67.5% |
| Llama-70B fp16, 8 replicas x 8 H100 | 6s | 2.339s | 0.290s | 87.6% |
| GLM-like 744B MoE, 8 replicas x 8 H100 | 6s | 1.313s | 0.005s | 99.6% |

This is exactly why we wanted a simulator. The point is not only that the idea
worked. The point is that we could sweep hardware shapes, routing policies,
prefix depth, and lead time quickly enough to find the boundary.

And there is a boundary.

On Llama-70B with 8 replicas x 4 A100s and only 6 seconds of prediction lead
time, warming a deep prefix was too slow. The p95 got worse. A shallow 16k-token
warm was roughly p95-neutral and improved mean TTFT by 10.5%. With 40 seconds
of lead time, full warming improved mean TTFT by 54.0%, but p95 only by 5.1%.

That result matters as much as the win. The simulator did not just say "cool
idea." It said:

```text
only warm a prefix depth that can finish before the burst
```

That is the kind of design rule you want before implementing a production
system.

## What The Simulator Is Good For

`inference-sim` is useful when you want to compare mechanisms, not when you want
to certify final production numbers.

Good uses:

- test a new routing policy against cache-aware and least-load baselines;
- replay trace windows under different cluster shapes;
- estimate whether prefix cache movement is worth it;
- compare H100, H200, B200, A100, or custom GPU specs;
- reason about TTFT versus throughput;
- produce reproducible figures for a systems blog or paper;
- find counterexamples before you overclaim.

Bad uses:

- claiming exact vendor benchmark numbers;
- replacing kernel-level profiling;
- ignoring real scheduler overheads;
- assuming all production traces have the same burst structure.

The simulator should be a microscope, not an oracle.

## Design Principles

We tried to keep the project opinionated in a few ways.

**All tunables should be visible.** Model size, active parameters, KV shape,
GPU specs, cache tiers, batch size, routing thresholds, trace offsets, and
arrival scaling all live in config or CLI flags.

**Policies should be pluggable.** A routing idea should be a small function
that can be compared against baselines on the same workload.

**The UI should make queues visible.** Looking at mean latency is not enough.
The useful questions are often visual: which node built a queue, which request
waited, and where did prefill block decode?

**Negative results should be easy to find.** A simulator is most useful when it
can disprove your favorite idea cheaply.

## Open-Source Launch

We are preparing `inference-sim` as an open-source project because inference
scheduling research needs more shared testbeds.

There are many excellent serving systems, but they are often too heavy for a
quick thought experiment. There are also many papers with bespoke simulators
that are hard to reproduce. We want something in the middle: small enough to
read, realistic enough to catch important tradeoffs, and easy to extend.

The first public version should include:

- CLI simulation runs;
- React UI for visual inspection;
- trace replay from public datasets;
- documented timing equations;
- routing baselines;
- predictive warming experiments;
- launch examples with saved JSON outputs.

The roadmap after launch:

- split weight dtype and KV dtype;
- add more trace adapters;
- add more serving policies;
- improve disaggregated cache modeling;
- add tests around timing equations and routing invariants;
- package reproducible benchmark scripts.

## Try It

Quickstart:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python server.py
```

Open:

```text
http://localhost:8000
```

Or run the CLI:

```bash
python3 simulate.py
```

The simulator is small on purpose. Read `config.py`, change one thing, rerun,
and see whether your intuition survives contact with a queue.

That is the whole point.

## Short Ending

Modern LLM inference is no longer just "make the model faster."

It is queues, caches, batches, memory, interconnects, workloads, and routing
policies all colliding at once.

`inference-sim` is our attempt to make those collisions visible.

If you are building an inference scheduler, testing a cache policy, or arguing
about where a request should go, we hope this gives you a cheap place to start.
