# Biting The Bullet: Predicting Inference Bursts And Warming KV Before The Queue Forms

Status: draft.

This post is based on simulator runs from `clean-experiments/results/`.
Numbers are not hardware measurements. They are meant to test the scheduling
mechanism and identify when it helps.

## Narrative Plan

1. Open with the core inference-serving dilemma: cache-aware routing saves
   prefill work, but can create queues during same-prefix bursts.
2. Explain TTFT, prefill, decode, queue time, and why long shared prefixes make
   this failure mode painful.
3. Set up the concrete workload: data labeling, agent fanout, tool calls, and
   batch jobs where hundreds of requests share the same long prefix.
4. Explain why the usual choices are incomplete:
   cache-only affinity queues; least-load routing recomputes; cache-aware
   routing reacts after the burst starts.
5. Introduce "bite the bullet": predict the burst and pay some prefill before
   requests arrive, so multiple replicas have useful KV when the burst lands.
6. Give the algorithmic rule: predict burst, estimate warm cost, warm only if
   the prefix depth can finish before the burst, otherwise warm less or skip.
7. Present the main synthetic result, in percentages.
8. Show the standard-system matrix to argue this is not only a GLM-specific
   result.
9. Show the negative/boundary results: A100 without enough lead time, Mooncake
   public trace, and reactive RDMA.
10. End with the real claim: this is a burst-aware primitive for a specific
    workload class, not a universal replacement for cache-aware routing.

## Draft

Large inference batches have a funny way of breaking otherwise reasonable
routing policies.

Imagine a data-labeling job, an agent spawning hundreds of subagents, or a tool
workflow that asks the model to score many records against the same long
document. The requests are not identical, but the first 64k tokens are the same.
Each request has a small unique suffix. Each request may only need one token of
output.

From the model's point of view, the expensive part is obvious: read the long
shared prefix once, cache the KV, and reuse it.

From the router's point of view, things are more awkward. If we route everything
to the replica that already has the prefix cached, that replica queues hundreds
of requests. If we spread requests by least load, we reduce queueing, but now
other replicas may have to redo the long prefill. The cache saved compute, but
the queue ate the latency.

This post is about a simple idea: what if we predict that burst before it
arrives, then "bite the bullet" early by prefilling the shared prefix on more
than one replica?

In other words: pay some prefill before the queue forms.

## The Two Parts Of LLM Inference

For user-visible latency, two metrics matter most.

**Time to first token, or TTFT**, is the time from request arrival until the
first output token appears. TTFT is usually dominated by:

- queue time: waiting behind other requests;
- prefill time: reading the input prompt and building KV cache.

**Per-token latency** is the time between streamed output tokens. That is mostly
decode speed: generating one token at a time.

For long-input, short-output workloads, TTFT is often the whole user
experience. If a labeling request has a 65k-token prompt and asks for a single
classification token, the request is basically all prefill.

That is the workload we care about here.

## The Burst Failure Mode

Suppose a batch job sends 500 requests in one second. Every request shares a
65,536-token prefix and has a 256-token unique suffix. The serving system has a
max decode batch of 256 sequences per replica.

There are three natural routing choices.

**Pure cache affinity** sends the request to the replica with the prefix cached.
That maximizes reuse, but it can pile the whole burst onto one replica.

**Least-load routing** spreads requests across replicas. That helps queueing,
but it may recompute the long shared prefix on replicas that do not have the
KV.

**Cache-aware routing** tries to balance both: prefer cache hits unless the
cached replica is too overloaded. This is a strong default, but it is still
reactive. It starts splitting after load has already appeared.

The idea here is to act before that point.

## Biting The Bullet

"Bite the bullet" is a predictive warming policy.

When a request arrives, or when job metadata appears, the system asks:

1. Does this look like the beginning of a large same-prefix burst?
2. Which prefix blocks are likely to be reused?
3. How much time do we have before the burst lands?
4. Which idle replicas can prefill part of that prefix in time?

Then it chooses an action:

- pin existing KV if a hot prefix is already resident;
- prefetch KV from host or disk if the cache exists outside HBM;
- replicate KV to another worker if fast interconnect is available;
- speculatively prefill the prefix on another replica if the KV does not exist.

The most important rule is not "always warm." The rule is:

> Only warm a prefix depth that can finish before the predicted burst. If that
> is not possible, warm a shallower prefix or do nothing.

That rule matters a lot in the results.

## Experiment Setup

The main synthetic workload is intentionally sharp:

- 8 burst jobs.
- 500 requests per burst.
- 65,536-token shared prefix.
- 256-token unique suffix.
- 1 output token.
- requests arrive within a 1 second burst window.
- no RDMA.
- HBM-only prefix cache.
- max serving batch: 256.
- cache-aware threshold: split when imbalance exceeds 8 in-flight requests.

The one-token output is deliberate. It models labeling, scoring, extraction,
and agent fanout where the answer is tiny and TTFT dominates.

The main GLM-like run uses:

- 8 independent replicas.
- each replica is 8 x H100 80GB with tensor parallelism 8.
- 64 total H100s.
- GLM-like 744B MoE shape.
- 40B active parameters per token.
- compressed MLA-like KV proxy.

Important caveat: the simulator currently has one dtype knob shared by weights
and KV. The GLM preset uses int4 weights and int4 KV as a proxy. Real fp8 KV
would make warmed-KV bytes about 2x larger than reported here.

## Main Result: Same-Prefix Burst

On the GLM-like 8 x 8 H100 setup, cache-aware routing gets p95 TTFT down to
1.313 seconds. Predictive warming improves that sharply.

| Warm depth | p95 TTFT | p95 reduction vs cache-aware | Mean TTFT reduction | Warm KV | Warm busy time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16,384 tokens | 0.982s | 25.2% | 18.6% | 11.8 GB | 10.6s |
| 32,768 tokens | 0.651s | 50.4% | 51.3% | 23.6 GB | 21.2s |
| 65,536 tokens | 0.005s | 99.6% | 99.5% | 47.1 GB | 42.4s |

This is the cleanest result in the project.

If the predictor is right and there is enough warm lead time, moving prefill
off the request critical path can almost erase TTFT for this workload. Partial
warming is the cheaper point: warming half the prefix cuts p95 TTFT by 50.4%
with half the warm KV and half the warm busy time of full warming.

Full warming is the latency winner when prediction is perfect.

## Is This Only A GLM Result?

To check that, we reran the same target workload across more standard simulated
systems.

| System | Lead time | Cache-aware p95 | Best warm p95 | p95 reduction | Mean TTFT reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-70B fp16, 4 replicas x 4 H100 | 6s | 6.333s | 1.638s | 74.1% | 84.4% |
| Llama-70B fp16, 8 replicas x 4 H100 | 6s | 5.047s | 1.638s | 67.5% | 82.2% |
| Llama-70B fp16, 8 replicas x 8 H100 | 6s | 2.339s | 0.290s | 87.6% | 93.3% |
| GLM-like 744B MoE, 8 replicas x 8 H100 | 6s | 1.313s | 0.005s | 99.6% | 99.5% |

That is the broader positive claim:

> On synchronized same-prefix bursts, speculative prefix warming reduces p95
> TTFT by 67-99% on H100-class simulated deployments.

This is not just a quirk of the GLM-like setup. The same mechanism helps across
Llama-70B/H100 systems too.

## But It Does Not Always Help

The A100 rows are the useful counterexample.

On Llama-70B fp16 with 8 replicas x 4 A100 and only 6 seconds of lead time,
warming 32k or 65k tokens is too slow. The speculative prefill is not ready
before the burst. Instead of helping, it makes p95 TTFT much worse:

| System | Lead time | Best tested warm depth | p95 result |
| --- | ---: | ---: | ---: |
| 8 replicas x 4 A100 | 6s | 32,768 tokens | 149.9% worse p95 |
| 8 replicas x 4 A100, shallow sweep | 6s | 16,384 tokens | 0.2% worse p95, 10.5% better mean |
| 8 replicas x 4 A100 | 40s | 65,536 tokens | 5.1% better p95, 54.0% better mean |

This is not a failure of the idea so much as a design constraint.

If warm time is longer than prediction lead time, do not warm that much. On
slower hardware, the policy should either warm a shallower prefix or skip
warming entirely.

The algorithm should be gated by a simple estimate:

```text
warm_time(model, hardware, prefix_depth) < predicted_lead_time - safety_margin
```

No inequality, no warming.

## What About Noisy Predictors?

We also added one-request decoy jobs and swept predictor precision from 1.0
down to about 0.25.

On the GLM-like 8 x 8 H100 setup, full warming still won latency even with
false positives:

| Warm depth | Actual precision | p95 TTFT | p95 reduction vs cache-aware | Warm KV |
| --- | ---: | ---: | ---: | ---: |
| 32,768 tokens | 1.00 | 1.176s | 31.9% | 23.6 GB |
| 32,768 tokens | 0.50 | 1.176s | 31.9% | 47.1 GB |
| 32,768 tokens | 0.25 | 1.076s | 37.7% | 94.2 GB |
| 65,536 tokens | 1.00 | 0.194s | 88.8% | 47.1 GB |
| 65,536 tokens | 0.50 | 0.176s | 89.8% | 94.2 GB |
| 65,536 tokens | 0.25 | 0.199s | 88.5% | 188.4 GB |

This was mildly surprising. The hypothesis going in was that full-prefix
warming would become fragile at moderate precision. In this roomy H100 setup,
it did not. False positives mostly showed up as extra warm cost, not worse p95.

That changes the claim. Partial warming is not always the latency optimum.
Partial warming is the cheaper operating point. Full warming is best when the
cluster has enough slack and predictor signals are good enough.

## Decode-Heavy Boundary

The short-output benchmark isolates TTFT. We also tried a decode-heavy variant
with 256 output tokens per request.

Cache-aware p95 TTFT was 1.728s. Full adaptive warming brought p95 TTFT down to
0.622s at perfect precision, a 64.0% reduction. At 0.5 precision, adaptive full
warming reached 0.587s p95, a 66.0% reduction.

The bad policy was confidence-scaled warming. At 0.5 precision it underwarmed
the true burst and hit 9.015s p95 TTFT. That is 421.7% worse than cache-aware.

The lesson: being "adaptive" is not automatically safe. The safe adaptation is
not arbitrary confidence scaling; it is warming only when there is enough idle
time to finish the chosen depth.

## Public Trace Boundary: Mooncake

The synthetic workload is the target workload. We also tested a public Mooncake
trace replay to check whether this pattern appears naturally in that trace.

Without RDMA, least-load-no-remote had p95 TTFT of 0.762s. Predictive fake
prefill was 0.795s, about 33ms worse. Real-seed warming was 0.794s, about 32ms
worse.

With 50 GB/s/GPU RDMA enabled, reactive copy was the best baseline at 0.695s
p95 TTFT. Prewarming plus reactive copy tied it at 0.695s. Fake prefill alone
was 0.795s, about 100ms worse.

That result is important:

> Mooncake replay did not validate the target workload by itself.

It does not seem to contain the large synchronized same-prefix bursts that make
this idea shine. And when fast reactive RDMA is available, that becomes a very
strong baseline.

## What This Means

"Bite the bullet" is not a universal routing replacement.

It is a burst-aware scheduling primitive for a specific but increasingly common
workload:

- data-labeling fanout;
- agent and subagent fanout;
- batch scoring against a shared document;
- multi-sample evaluation with a shared system prompt;
- tool workflows where metadata reveals a batch before every request arrives.

The positive result is strong: on H100-class simulated deployments, the target
workload sees 67-99% p95 TTFT reductions.

The boundary is also clear: the policy needs enough prediction lead time and
enough idle/slack capacity. If warming cannot finish before the burst, warming
can be worse than doing nothing.

That gives a clean production policy:

1. detect likely same-prefix burst;
2. estimate burst size and prefix depth;
3. estimate warm time on each candidate replica;
4. warm the deepest prefix that can finish before the burst;
5. route burst requests across replicas with warm KV;
6. fall back to normal cache-aware routing when confidence, lead time, or slack
   is insufficient.

The short version:

> Cache-aware routing reacts to the queue. Biting the bullet tries to move the
> expensive prefill before the queue exists.

That is the bet.

## Future Work

The next experiments should make the claim harder to dismiss:

- split weight dtype and KV dtype in the simulator, so int4 weights with fp8 KV
  are modeled directly;
- add a predictor trained on job metadata, not just synthetic oracle signals;
- test real data-labeling or agent traces with actual synchronized fanout;
- add an online controller that chooses prefix depth from lead-time estimates;
- compare against reactive RDMA, prefetch-from-host, and disaggregated cache
  systems under the same workload.

The strongest paper/blog version is not "we found a trick that always wins."
It is:

> In bursty same-prefix workloads, cache locality and load balancing are not
> enough. If the burst is predictable, speculative prefill can turn idle time
> before the burst into lower p95 TTFT during the burst.
