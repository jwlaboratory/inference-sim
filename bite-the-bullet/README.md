# Bite The Bullet

This folder contains a focused experiment for predictive KV warming:

1. A seed request warms a shared prefix on one worker.
2. A burst of same-prefix requests arrives later.
3. The runner compares:
   - cache-only affinity, which keeps sending the burst to the hot worker
   - least-load routing, which spreads requests but recomputes the prefix
   - reactive RDMA copy, which copies KV only once the cold worker is chosen
   - predictive RDMA copy, which copies KV before the burst
   - predictive fake-prefill, which warms another worker by spending prefill compute early

Run it from the repo root:

```bash
python3 bite-the-bullet/run.py
```

Useful knobs:

```bash
python3 bite-the-bullet/run.py --prefix-tokens 16384 --burst 256 --lead 1.0
python3 bite-the-bullet/run.py --rdma-gbps 12.5
python3 bite-the-bullet/run.py --skip-no-rdma
```

## ART Burst Predictor

Train a lightweight predictor on real ART-Chat rows:

```bash
python3 bite-the-bullet/predict_bursts_art.py
```

The label is intentionally cache-specific: a request is positive if at
least `K` later requests within `H` seconds share the same first `N`
prefix-cache blocks. The script saves a pure-Python logistic regression to
`bite-the-bullet/art_burst_model.json` and compares it with a simple
same-prefix momentum rule.

Useful knobs:

```bash
python3 bite-the-bullet/predict_bursts_art.py --key-blocks 4 --horizon-s 20 --future-k 2
python3 bite-the-bullet/predict_bursts_art.py --windows 12 --rows-per-window 1200 --train-windows 8
```

## End-To-End ART Warming

Evaluate whether predictor-triggered warming is net better in the simulator:

```bash
python3 bite-the-bullet/evaluate_art_warming.py
```

The evaluator compares cache/load baselines against background RDMA copy and,
optionally, fake-prefill warming:

```bash
python3 bite-the-bullet/evaluate_art_warming.py --warm-blocks 8 32 --thresholds 0.695
python3 bite-the-bullet/evaluate_art_warming.py --rdma-gbps 0 --include-fake-prefill
python3 bite-the-bullet/evaluate_art_warming.py --rdma-gbps 0 --hbm-only
python3 bite-the-bullet/evaluate_art_warming.py --arrival-scale 0.25
python3 bite-the-bullet/evaluate_art_warming.py --score-mode oracle --rdma-gbps 0 --hbm-only
```

Mooncake trace examples:

```bash
python3 bite-the-bullet/predict_bursts_art.py \
  --dataset valeriol29/mooncake-traces --config-name mooncake \
  --block-tokens 512 --key-blocks 8 --model-out bite-the-bullet/mooncake_burst_model.json

python3 bite-the-bullet/evaluate_art_warming.py \
  --model bite-the-bullet/mooncake_burst_model.json \
  --dataset valeriol29/mooncake-traces --config-name mooncake \
  --block-tokens 512 --key-blocks 8 --rdma-gbps 0 --hbm-only --include-real-seed
```

See `RESULTS.md` for the Mooncake HBM-only cache-aware threshold sweep.
