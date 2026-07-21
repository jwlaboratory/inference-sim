# Infer-Sim: An Open-Source Simulator For Routing Algorithms And Cache Policies For Inference Workloads

Status: draft.

Repo name: `inference-sim`.

## TLDR

Inference optimization is painful because many ideas cannot be tested cleanly
without production-like traffic. Logs tell you what happened, but they do not
make it easy to ask what would have happened under a different router, cache
policy, batch size, model, or GPU cluster.

Infer-Sim is a lightweight open-source simulator for replaying inference traces
through configurable routing policies, prefix-cache tiers, batching behavior,
and GPU clusters. The goal is to make inference-scheduling ideas cheap to test,
easy to visualize, and easier to argue about.

## Motivation

One of the biggest problems I noticed during my time at Morph was how difficult
it was to test theories about inference optimizations.

Testing a new idea often meant shipping code, watching production dashboards,
and waiting days before knowing whether the change helped. Even then, it was
hard to know why something improved or regressed. Raw logs are difficult to
visualize. A queue build-up can hide inside aggregate latency. A p95 TTFT spike
can come from prefill, decode, routing, cache misses, or plain old overload.

And inference systems are full of knobs.

The same router can behave differently depending on:

- the arrival pattern of live traffic;
- prompt length and output length distributions;
- model size and active parameters;
- KV cache size;
- quantization;
- batch size;
- GPU memory bandwidth;
- interconnect bandwidth;
- cache locality;
- whether the cache lives in HBM, host RAM, disk, or another node.

That makes backtesting hard. Production traffic is often the only thing that
stresses your system in the right way, but production is the worst place to
debug a scheduling theory from scratch.

Infer-Sim is an attempt to make that loop shorter.

Instead of immediately pushing a routing change to production, you can replay a
trace, change the router, change the GPU cluster, change the model, change the
cache policy, and inspect what happens.

It is not a replacement for real benchmarking. It is a way to test mechanisms
before they become distributed-systems projects.

## What You Can Tune

Infer-Sim replays inference requests through a configurable cluster. The current
simulator lets you tune:

- Mooncake-compatible trace dataset;
- request arrival speed/frequency;
- model parameters;
- active parameters per token;
- layers;
- KV heads and head dimension;
- dtype/quantization proxy;
- max serving batch size;
- router policy;
- cache-aware routing thresholds;
- GPU type;
- number of nodes;
- GPUs per node;
- FLOPs;
- HBM bandwidth and capacity;
- host RAM bandwidth;
- RDMA bandwidth;
- disk bandwidth;
- prefix-cache block size;
- HBM/RAM/disk cache behavior.

The goal is not to hide the knobs. The goal is to make them explicit.

## What You Can Visualize

The simulator has both a CLI and a small UI. You can inspect:

- mean latency;
- p95 latency;
- mean TTFT;
- p95 TTFT;
- cache-hit rate;
- node utilization;
- per-node request routing;
- backlog queues;
- peak queue size;
- request timelines;
- prefill and decode spans;
- where routing decisions created or avoided queues.

The visualization matters because averages are often too polite.

A routing policy can improve mean latency while quietly creating a tail problem
on one node. Or a cache policy can increase hit rate while still hurting TTFT
because it sends too many requests to the same replica.

Seeing the queue is different from reading a table.

## What The Simulator Models

Infer-Sim is small on purpose. The core engine is just a handful of files:

- `config.py`: all tunables;
- `workload.py`: trace loading and request construction;
- `gpu.py`: GPU timing equations and prefix-cache behavior;
- `router.py`: routing policies;
- `simulate.py`: event loop;
- `server.py`: API for the UI;
- `ui/`: React UI for interactive runs.

The simulator models:

- continuous batching;
- prefill timing;
- decode timing;
- model weight memory;
- KV cache memory;
- prefix block reuse;
- HBM cache;
- host RAM cache;
- disk cache;
- remote peer cache over RDMA;
- tensor-parallel nodes;
- independent serving replicas;
- pluggable routing policies.

The built-in routing policies include cache-aware and least-load baselines, but
the point is to make it easy to add your own.

## How We Approximate Each Variable

The simulator uses simple, inspectable equations. That is a feature. If a number
looks surprising, you should be able to trace where it came from.

### Requests

Requests are replayed from a Mooncake-compatible trace format. Each row includes
arrival time, input length, output length, and prefix block hashes.

The arrival gaps can be scaled to stress the system:

```text
arrival_time = recorded_arrival_time * ARRIVAL_SCALE
```

Lower arrival scale means a hotter replay. Higher arrival scale means a calmer
replay.

### Prefix Cache

Prompts are represented as blocks. Two requests share a prefix when their
leading block hashes match.

This lets the simulator model exact prefix-cache reuse without tokenizing the
text again.

For each request, the simulator finds the longest cached prefix available from:

- local HBM;
- local host RAM;
- peer node over RDMA;
- disk.

It uses the cached prefix only when loading it is faster than recomputing it.

### Prefill

Prefill is modeled as compute-bound:

```text
prefill_time = 2 * active_params * tokens / (flops * MFU)
```

This captures the intuition that long prompts are expensive because the model
has to process every input token.

### Decode

Decode is modeled as memory-bound across the active batch:

```text
decode_step_time =
  (active_weight_bytes + batch_kv_bytes) / (hbm_bandwidth * MBU)
```

The active weights are read once for the batch, while each sequence contributes
KV traffic.

### Cache Movement

Cache movement is modeled as bandwidth-bound:

```text
cache_load_time = kv_bytes / tier_bandwidth
```

Different tiers use different bandwidths:

- HBM is effectively local;
- host RAM uses PCIe bandwidth;
- peer cache uses RDMA bandwidth;
- disk uses local disk bandwidth.

### GPU Cluster

Each node is a group of GPUs serving together with tensor parallelism. Compute,
HBM bandwidth, HBM capacity, RAM bandwidth, RDMA bandwidth, and disk bandwidth
are aggregated across the GPUs in the node.

Nodes are independent serving replicas. Each node must fit the model in its
combined HBM.

### Queueing And Batching

Each node has a queue. Requests wait until they can be admitted. Decode runs as
a continuous batch up to `MAX_BATCH`.

Prefill pauses decode in the current model, which approximates
prefill-prioritizing schedulers and makes prefill/queue interactions visible.

## Why This Is Useful

The simulator is useful for questions like:

- Should this workload use cache-aware routing or least-load routing?
- How sensitive is p95 TTFT to the cache-aware threshold?
- Does a bigger batch help throughput or just hide queueing?
- When is remote KV faster than recomputing?
- When does cache affinity hurt tail latency?
- How much HBM does a prefix-cache policy consume?
- What happens if the same workload runs on H100s instead of A100s?
- How much prediction lead time does a speculative policy need?

These are not questions you want to answer by guessing.

## Case Study: Biting The Bullet

The first research project we built on top of Infer-Sim was **Biting the
Bullet**.

The idea is simple: when a large batch of same-prefix requests is about to
arrive, predict it and speculatively prefill the shared prefix on multiple
replicas before the queue forms.

This targets workloads like:

- data-labeling fanout;
- agent/subagent fanout;
- scoring many records against one shared document;
- batch jobs with shared system/tool prefixes.

The clean synthetic workload:

- 8 burst jobs;
- 500 requests per burst;
- 65,536-token shared prefix;
- 256-token unique suffix;
- 1 output token;
- 1 second burst window;
- no RDMA;
- HBM-only cache;
- max serving batch: 256.

On a GLM-like 744B MoE proxy with 8 replicas x 8 H100s, cache-aware routing got
p95 TTFT to 1.313 seconds. Predictive warming reduced it sharply:

| Warm depth | p95 TTFT | p95 reduction vs cache-aware | Warm KV | Warm busy time |
| --- | ---: | ---: | ---: | ---: |
| 16,384 tokens | 0.982s | 25.2% | 11.8 GB | 10.6s |
| 32,768 tokens | 0.651s | 50.4% | 23.6 GB | 21.2s |
| 65,536 tokens | 0.005s | 99.6% | 47.1 GB | 42.4s |

We also tested the same burst on more standard Llama-70B/H100 simulated systems:

| System | Lead time | Cache-aware p95 | Best warm p95 | p95 reduction |
| --- | ---: | ---: | ---: | ---: |
| Llama-70B fp16, 4 replicas x 4 H100 | 6s | 6.333s | 1.638s | 74.1% |
| Llama-70B fp16, 8 replicas x 4 H100 | 6s | 5.047s | 1.638s | 67.5% |
| Llama-70B fp16, 8 replicas x 8 H100 | 6s | 2.339s | 0.290s | 87.6% |
| GLM-like 744B MoE, 8 replicas x 8 H100 | 6s | 1.313s | 0.005s | 99.6% |

That gave us a strong but scoped result:

> On synchronized same-prefix bursts, speculative prefix warming can reduce p95
> TTFT by 67-99% on H100-class simulated deployments.

But the simulator also found the boundary.

On Llama-70B with 8 replicas x 4 A100s and only 6 seconds of lead time, warming
a deep prefix was too slow and p95 got worse. A shallow 16k-token warm was
roughly p95-neutral and improved mean TTFT by 10.5%. With 40 seconds of lead
time, full warming improved mean TTFT by 54.0%, but p95 only by 5.1%.

This is exactly what we want from a simulator: not just a positive result, but
the rule around it.

For this policy, the rule is:

```text
only warm a prefix depth that can finish before the predicted burst
```

Without a simulator, that kind of boundary is easy to miss.

## What Infer-Sim Is Not

Infer-Sim is not a production benchmark.

It does not replace:

- profiling a real serving stack;
- measuring kernel overheads;
- testing on real hardware;
- validating against live traffic;
- understanding scheduler-specific implementation details.

The simulator is for mechanism exploration. It helps answer:

> Is this idea plausible enough to implement?

It should not be used to claim exact production latency.

## Try It

GitHub:

```text
https://github.com/jwlaboratory/inference-sim
```

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

The first run replays a public trace window. From there, edit `config.py` or use
the UI to change model shape, GPU cluster, batch size, arrival rate, and routing
policy.

## Feedback And Extensions

Please try it and share feedback. We would love extensions to the open-source
repository, especially:

- new routing policies;
- new trace adapters;
- better cache policies;
- better disaggregated-cache modeling;
- more GPU specs;
- more visualizations;
- validation against real serving measurements.

The broader hope is that inference-scheduling ideas become easier to test and
harder to overclaim.

Logs are where you see what happened.

Infer-Sim is where you can ask what would have happened.
