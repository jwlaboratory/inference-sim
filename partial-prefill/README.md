# Partial Prefill

This experiment tests a softer version of predictive KV warming:

1. Predict a large same-prefix burst.
2. Fake-prefill only the first `N` tokens of the shared prompt on extra workers.
3. Route the burst across workers that have this partial prefix.
4. Let each real request recompute the remaining prompt.

The idea is to trade some upside for much lower false-positive cost. Full
fake-prefill is best when the predictor is very accurate; partial fake-prefill
may be better when the predictor is noisy or when the prefix is very long.

Run from the repo root:

```bash
python3 partial-prefill/sweep_partial_prefill.py
```

The default setup is the synthetic data-labeling spike benchmark:

- 8 true jobs x 500 requests/job
- 65,536 shared prefix tokens/job
- 120 decoy long-prefix jobs
- 1 output token/request
- no RDMA, HBM-only
- stubborn cache-aware baseline with `imbalance_abs=512`
- predictor recall fixed at 1.0 while precision is swept

See `RESULTS.md` for the current `abs=64` and `abs=512` sweep readout.
