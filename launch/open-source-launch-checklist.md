# Open-Source Launch Checklist

Status: draft.

## Positioning

- Project name: `inference-sim`.
- One-line description:
  "A small, inspectable simulator for LLM inference queues, prefix caches,
  routing policies, and GPU cluster tradeoffs."
- Primary audience:
  inference engineers, systems researchers, founders building serving stacks,
  and people evaluating routing/cache ideas.
- Main promise:
  test scheduling mechanisms cheaply before implementing them in a production
  serving system.
- Main caveat:
  simulator numbers are mechanism-validation results, not hardware benchmarks.

## Must Do Before Public Launch

- Choose a license.
  - Recommended default if you want permissive adoption: Apache-2.0.
  - Alternative: MIT if you want the simplest permissive license.
  - Do not launch without an explicit license.
- Add `LICENSE`.
- Add `CONTRIBUTING.md`.
- Add `CODE_OF_CONDUCT.md` or explicitly decide not to.
- Add `SECURITY.md` with a contact path for vulnerability reports.
- Add `CITATION.cff` if we want researchers to cite the simulator.
- Confirm repository has no secrets or local paths in committed files.
- Decide whether generated result JSONs should stay in-repo or move to a
  release artifact.
- Add screenshots or a short GIF of the UI.
- Add a simple architecture diagram to the README.
- Add a "known limitations" section to the README.
- Add a "reproduce the blog numbers" section with exact commands.
- Add basic smoke tests for:
  - model-fit validation;
  - prefix-cache hit/miss behavior;
  - routing thresholds;
  - deterministic replay with a fixed seed.
- Add CI that runs:
  - Python compile check;
  - tests;
  - optional UI build.
- Pin Python dependency versions more tightly than the current
  `requirements.txt`.
- Confirm Hugging Face dataset access path works from a fresh clone.
- Add fallback instructions for users without network access to the datasets
  API.

## Nice To Have Before Launch

- Dockerfile or `uv` setup.
- `make demo`, `make test`, `make ui`.
- Example configs:
  - Llama-70B on 4 x H100 replicas;
  - Llama-70B on 8 x H100 replicas;
  - GLM-like MoE proxy;
  - no RDMA versus RDMA.
- Saved small demo trace so the first run does not depend on a remote dataset.
- More comments around timing equations in `gpu.py`.
- A small `examples/` folder with:
  - cache-aware baseline;
  - remote KV copy;
  - predictive warming;
  - standard-system matrix.
- Add GitHub issue templates:
  - bug report;
  - policy idea;
  - trace adapter request.
- Add PR template.
- Add repo topics:
  - `llm-inference`
  - `inference-serving`
  - `prefix-cache`
  - `gpu-scheduling`
  - `simulator`
  - `kv-cache`

## README Shape

The public README should answer, in order:

1. What is this?
2. Why should I care?
3. Quickstart.
4. What does it model?
5. What does it not model?
6. Example result.
7. How to add a routing policy.
8. How to reproduce launch/blog experiments.
9. Project roadmap.
10. Citation/license/contributing.

Suggested opening:

> `inference-sim` is a small simulator for LLM inference clusters. It replays
> request traces through configurable GPU replicas, prefix-cache tiers, and
> routing policies so you can reason about TTFT, queueing, continuous batching,
> and KV reuse before implementing a serving-system change.

## Launch Assets

- Blog 1: predictive warming research post:
  `https://github.com/shreybirmiwal/bite-the-bullet/blob/main/BLOG.md`.
- Blog 2: simulator launch post:
  `launch/open-source-engine-blog-draft.md`.
- Results:
  `https://github.com/shreybirmiwal/bite-the-bullet/blob/main/clean-experiments/results/SUMMARY.md`.
- Cross-system matrix:
  `https://github.com/shreybirmiwal/bite-the-bullet/blob/main/clean-experiments/results/standard-systems/SUMMARY.md`.
- Suggested launch screenshot:
  UI showing a Gantt chart with cache-aware versus predictive warming.
- Suggested launch figure:
  p95 TTFT reduction table across H100 systems.

## Launch Sequence

1. Finish README and license.
2. Add tests and CI.
3. Create a small demo dataset path or fixture.
4. Re-run the research experiments from fresh `inference-sim` and
   `bite-the-bullet` clones.
5. Tag `v0.1.0`.
6. Publish simulator launch blog.
7. Publish predictive warming blog.
8. Post announcement copy.
9. Collect issues and examples from first users.

## Claims To Make

Safe claims:

- "The simulator models queues, continuous batching, prefix-cache reuse, and
  cache movement across configurable GPU clusters."
- "It is designed for mechanism exploration, not vendor benchmarking."
- "On our synchronized same-prefix burst benchmark, speculative warming reduced
  p95 TTFT by 67-99% on H100-class simulated deployments."
- "The same experiments expose a boundary condition: warming can be harmful if
  it cannot finish before the burst."

Claims to avoid:

- "This predicts production latency exactly."
- "Predictive warming always beats cache-aware routing."
- "The simulator is a full replacement for benchmarking on real hardware."
- "Public traces universally contain the target burst pattern."

## Open Questions

- Which license do we want?
- Should result JSONs live in git, Git LFS, or GitHub releases?
- Do we want to rename the project before launch?
- Should the launch repo include the research experiments by default, or should
  those live in a separate branch/folder?
- What is the minimal CI we want before telling people to try it?
