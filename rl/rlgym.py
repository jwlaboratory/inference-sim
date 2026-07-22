"""RL gym for the router: trace-replay episodes + cross-entropy-method training.

An episode replays one trace window through simulate.run with the learned
policy's weight vector W swapped in; the return is a latency-based cost.
Training runs CEM (a simple, robust evolution strategy): sample a population
of W around the current mean, keep the elites, refit, repeat. Deterministic
given the trace, so improvements are real policy improvements.

Objective per window: 0.7 * mean_lat + 0.3 * p95_lat, normalized by the
least_load baseline on that window so easy and hard windows weigh equally.
Weights are selected on held-out windows the trainer never fits.

Run:  python3 rl/rlgym.py            # train, then eval train + holdout windows
      python3 rl/rlgym.py eval       # eval current learned-policy W only
"""
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import simulate
import workload
from rl.learned import Learned

TRAIN_SEEDS = [42, 7, 123, 9, 77]
HOLDOUT_SEEDS = [5, 314, 2024]
POP, ELITE, ITERS = 24, 6, 30
INIT_STD, MIN_STD, SMOOTH = 0.5, 0.05, 0.7
NAMES = ["wait_prefill", "wait_decode", "run_decode", "n_wait", "n_run",
         "marg_prefill", "local_hit", "remote_hit", "kv_used",
         "momentum", "burst_x_cache"]
WEIGHTS_PATH = Path(__file__).with_name("rlgym_weights.json")


def load_windows(seeds):
    wins = []
    for seed in seeds:
        d = config.as_dict()
        d["SEED"] = seed
        cfg = SimpleNamespace(**d)
        try:
            reqs = workload.generate(cfg)
            simulate.run("least_load", reqs, cfg)   # validate + baseline
        except ValueError as e:
            print(f"  skipping window seed={seed}: {e}")
            continue
        wins.append((seed, cfg, reqs))
    return wins


def episode(w, cfg, reqs):
    """One rollout of weight vector w; returns the run's metrics."""
    Learned.W = w
    random.seed(0)                       # freeze any tie-break randomness
    return simulate.run("learned", reqs, cfg)["metrics"]


def cost(m, base):
    return 0.7 * m["mean_lat"] / base["mean_lat"] + \
           0.3 * m["p95_lat"] / base["p95_lat"]


def evaluate(w, windows, baselines):
    return sum(cost(episode(w, cfg, reqs), baselines[seed])
               for seed, cfg, reqs in windows) / len(windows)


def baselines_for(windows):
    out = {}
    for seed, cfg, reqs in windows:
        random.seed(0)
        out[seed] = simulate.run("least_load", reqs, cfg)["metrics"]
    return out


def train():
    windows = load_windows(TRAIN_SEEDS)
    holdout = load_windows(HOLDOUT_SEEDS)
    base_t, base_h = baselines_for(windows), baselines_for(holdout)

    dim = len(Learned.W)
    mean = list(Learned.W)               # start from the cost-model prior
    std = [INIT_STD] * dim
    rng = random.Random(1)
    best_w, best_hold = list(mean), float("inf")

    for it in range(ITERS):
        pop = [[rng.gauss(m, s) for m, s in zip(mean, std)] for _ in range(POP)]
        pop[0] = list(mean)              # always keep the incumbent
        scored = sorted((evaluate(w, windows, base_t), w) for w in pop)
        elites = [w for _, w in scored[:ELITE]]
        mean = [SMOOTH * sum(e[i] for e in elites) / ELITE + (1 - SMOOTH) * mean[i]
                for i in range(dim)]
        std = [max(SMOOTH * (sum((e[i] - mean[i]) ** 2 for e in elites)
                             / ELITE) ** 0.5 + (1 - SMOOTH) * std[i], MIN_STD)
               for i in range(dim)]
        hold = evaluate(scored[0][1], holdout, base_h)
        if hold < best_hold:             # model selection on held-out windows
            best_hold, best_w = hold, list(scored[0][1])
        print(f"iter {it:2d}  train {scored[0][0]:.4f}  holdout {hold:.4f}  "
              f"best_holdout {best_hold:.4f}")

    print("\nbest W (selected on holdout):")
    for n, v in zip(NAMES, best_w):
        print(f"  {n:<14} {v:8.3f}")
    with WEIGHTS_PATH.open("w") as f:
        json.dump({"W": best_w, "holdout_cost": best_hold}, f, indent=1)
    return best_w, windows, holdout


def report(windows, label):
    print(f"\n=== {label} ===")
    for seed, cfg, reqs in windows:
        print(f"window seed={seed}")
        for pol in ("cache_aware", "least_load", "learned"):
            random.seed(0)
            m = simulate.run(pol, reqs, cfg)["metrics"]
            print(f"  {pol:<12} mean {m['mean_lat']:6.2f}  p95 {m['p95_lat']:6.2f}  "
                  f"ttft {m['mean_ttft']:5.2f}  peakq {m['peak_queue']:6.2f}  "
                  f"hbm {m['hbm_hit']:4.0%}  util {m['util']:4.0%}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        report(load_windows(TRAIN_SEEDS), "train windows")
        report(load_windows(HOLDOUT_SEEDS), "holdout windows")
    else:
        w, windows, holdout = train()
        Learned.W = w
        report(windows, "train windows")
        report(holdout, "holdout windows")
