# Announcement Copy

## GitHub Repository Description

A small simulator for LLM inference queues, prefix caches, routing policies,
continuous batching, and GPU cluster tradeoffs.

## Short Social Post

We are open-sourcing `inference-sim`, a small simulator for LLM inference
scheduling.

It replays request traces through configurable GPU clusters and models
continuous batching, prefill/decode timing, prefix-cache reuse, HBM/RAM/disk
cache tiers, RDMA, and routing policies.

The goal is simple: test inference scheduling ideas before turning them into a
distributed systems project.

First case study: predictive prefix warming for synchronized same-prefix
bursts. In simulation, it cuts p95 TTFT by 67-99% on H100-class setups, while
also exposing where the idea fails.

## Longer Social Post

LLM inference routing is hard to reason about from intuition alone.

Cache affinity can save prefill but build queues. Least-load routing can reduce
queues but recompute long prefixes. Cache-aware routing is better, but still
depends heavily on workload shape, model size, GPU memory, and interconnect.

We built `inference-sim` as a small open-source simulator for these tradeoffs.

It models:

- continuous batching;
- prefill and decode timing;
- prefix-cache hits and misses;
- HBM, host RAM, disk, and peer-node cache tiers;
- configurable GPU clusters;
- pluggable routing policies.

The first research use case is "bite the bullet": predicting large same-prefix
bursts and speculatively prefilling KV on multiple replicas before the queue
forms.

On H100-class simulated systems, this reduces p95 TTFT by 67-99% for the target
same-prefix burst benchmark. On slower A100 systems without enough prediction
lead time, it can hurt, which is exactly the kind of boundary we wanted the
simulator to reveal.

This is not a replacement for real benchmarking. It is a cheap place to test
mechanisms, find counterexamples, and make inference-scheduling arguments more
reproducible.

## Hacker News / Reddit Style Post

We built a small simulator for LLM inference scheduling.

The motivation: routing decisions in inference clusters are coupled to prefix
cache state, queue depth, prefill cost, decode batching, HBM capacity, and
interconnect speed. It is easy to argue from intuition and hard to know which
effect dominates.

`inference-sim` replays request traces through configurable GPU replicas. It
models continuous batching, prefill/decode timing, prefix-cache reuse, cache
movement through HBM/RAM/disk/RDMA tiers, and pluggable routing policies.

The simulator is intentionally not a production benchmark. It is for mechanism
exploration.

The first case study is speculative prefix warming. If a data-labeling or agent
job is about to send hundreds of requests with the same long prefix, cache-aware
routing can still form queues. We simulate predicting the burst and prefilling
the prefix on multiple replicas before it arrives. On H100-class simulated
systems, this cuts p95 TTFT by 67-99% on the target burst benchmark. On A100
without enough lead time, it fails, which gives a useful policy rule: only warm
a prefix depth that can finish before the burst.

Would love feedback from people building serving systems, routers, cache
policies, or inference benchmarks.

## Blog Cross-Link

For the simulator:

> We built `inference-sim` because scheduling ideas are too expensive to test
> only after implementing them in a serving stack.

For the predictive warming post:

> Once we had the simulator, we used it to study a specific failure mode:
> synchronized same-prefix bursts that force a router to choose between cache
> locality and load balancing.

## Release Notes Draft

`inference-sim` v0.1.0 includes:

- trace replay for LLM inference workloads;
- configurable model and GPU specs;
- continuous batching simulator;
- prefix-cache tiers for HBM, RAM, disk, and peer-node RDMA;
- cache-aware and least-load routing baselines;
- React UI for interactive simulation;
- CLI simulation entrypoints;
- predictive KV warming experiments and saved results.
