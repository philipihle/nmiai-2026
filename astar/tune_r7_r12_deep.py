"""Deep parameter search for R7 and R12 — the worst-scoring rounds.

Extended search space including port_prob, death_competition_factor,
food_competition, and food_pop_drain ranges.

Usage:
  python tune_r7_r12_deep.py
  python tune_r7_r12_deep.py --rounds 7    # only R7
  python tune_r7_r12_deep.py --mc-deep 500 # more MC for deep eval
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import time

import numpy as np

from backtest import score_prediction
from simulator import AstarSimulator, SimParams, _grid_to_counts

CACHE_DIR = "replay_cache"

_worker_seeds_data = None


def _init_worker(seeds_data):
    global _worker_seeds_data
    _worker_seeds_data = seeds_data


def _eval_combo(params_dict):
    params = SimParams(**params_dict)
    scores = []
    for sd in _worker_seeds_data:
        sim = AstarSimulator(sd["grid"], sd["settlements"], params=params, seed=123)
        probs = sim.monte_carlo(n_runs=20, n_steps=50, n_workers=1)
        score = score_prediction(probs, sd["gt"])
        scores.append(score)
    return (params_dict, float(np.mean(scores)), scores)


def _eval_combo_deep(params_dict):
    mc_runs = params_dict.pop("__mc_runs", 500)
    params = SimParams(**params_dict)
    scores = []
    for sd in _worker_seeds_data:
        sim = AstarSimulator(sd["grid"], sd["settlements"], params=params, seed=456)
        probs = sim.monte_carlo(n_runs=mc_runs, n_steps=50, n_workers=1)
        score = score_prediction(probs, sd["gt"])
        scores.append(score)
    return (params_dict, float(np.mean(scores)), scores)


def load_round_seeds(rn):
    detail_path = os.path.join(CACHE_DIR, f"round{rn}_detail.json")
    if not os.path.exists(detail_path):
        return [], 0.0
    with open(detail_path) as f:
        detail = json.load(f)
    seeds = []
    rates = []
    for si in range(detail.get("seeds_count", 5)):
        replay_path = os.path.join(CACHE_DIR, f"round{rn}_seed{si}_replay.json")
        analysis_path = os.path.join(CACHE_DIR, f"round{rn}_seed{si}_analysis.json")
        if not os.path.exists(replay_path) or not os.path.exists(analysis_path):
            continue
        with open(replay_path) as f:
            replay = json.load(f)
        with open(analysis_path) as f:
            analysis = json.load(f)
        gt = np.array(analysis["ground_truth"])
        settle_rate = float(gt[:, :, 1].mean() + gt[:, :, 2].mean())
        rates.append(settle_rate)
        seeds.append({
            "round": rn, "seed": si,
            "grid": np.array(detail["initial_states"][si]["grid"]),
            "settlements": replay["frames"][0]["settlements"],
            "gt": gt,
        })
    return seeds, np.mean(rates) if rates else 0.0


def build_extended_combos():
    """Extended parameter space for R7/R12 overshoot problem."""
    grid = {
        "spawn_prob":        [0.05, 0.06, 0.08, 0.10, 0.12],
        "spawn_pop_threshold": [0.3, 0.5, 0.7],
        "death_base_rate":   [0.005, 0.01, 0.02, 0.03, 0.04, 0.05],
        "death_food_factor": [0.02, 0.04, 0.06, 0.08, 0.10],
        "pop_growth_rate":   [0.06, 0.08, 0.10, 0.12],
        "food_base_regen":   [0.15, 0.20, 0.25],
        "food_competition":  [0.02, 0.03, 0.04, 0.05, 0.06, 0.08],
        "food_pop_drain":    [0.06, 0.08, 0.10, 0.12],
        "food_forest_bonus": [0.02, 0.04, 0.06],
        "port_prob":         [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
        "death_competition_factor": [0.01, 0.02, 0.03, 0.04, 0.05],
    }

    fixed = dict(
        spawn_max_per_step=40,
        port_wealth_threshold=0.0,
        wealth_decay=0.0,
        wealth_coastal_rate=0.01,
        port_survival_bonus=0.0,
    )

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    # Full grid is too large — sample randomly
    total_full = 1
    for v in values:
        total_full *= len(v)
    print(f"Full grid: {total_full:,} combos")

    max_combos = 50000
    combos = []
    if total_full <= max_combos:
        for vals in itertools.product(*values):
            d = dict(zip(keys, vals))
            d.update(fixed)
            combos.append(d)
    else:
        # Random sample
        rng = np.random.RandomState(42)
        seen = set()
        while len(combos) < max_combos:
            vals = tuple(rng.choice(grid[k]) for k in keys)
            if vals not in seen:
                seen.add(vals)
                d = dict(zip(keys, vals))
                d.update(fixed)
                combos.append(d)
        print(f"Sampled {len(combos)} combos")

    return combos, keys


def tune_round(rn, n_cpus, mc_deep=500):
    seeds, settle_rate = load_round_seeds(rn)
    if not seeds:
        print(f"R{rn}: No data")
        return None

    print(f"\n{'='*60}")
    print(f"R{rn}: settle_rate={settle_rate:.3f}, {len(seeds)} seeds")
    print(f"{'='*60}")

    combos, keys = build_extended_combos()
    total = len(combos)
    print(f"{total} combos x {len(seeds)} seeds x 20 MC (coarse)")

    # Coarse
    t0 = time.time()
    results = []
    with mp.Pool(n_cpus, initializer=_init_worker, initargs=(seeds,)) as pool:
        for i, result in enumerate(pool.imap_unordered(_eval_combo, combos), 1):
            results.append(result)
            if i % (n_cpus * 2) == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                best = max(r[1] for r in results)
                print(f"  [{i:>6d}/{total}] {i*100/total:.0f}%  {elapsed:.0f}s  "
                      f"ETA {eta:.0f}s  best={best:.1f}", flush=True)

    results.sort(key=lambda x: -x[1])

    # Deep eval top 10
    top_n = min(10, len(results))
    top = [r[0] for r in results[:top_n]]
    for t in top:
        t["__mc_runs"] = mc_deep

    print(f"\nDeep eval top {top_n} ({mc_deep} MC)...")
    deep_results = []
    with mp.Pool(n_cpus, initializer=_init_worker, initargs=(seeds,)) as pool:
        for i, result in enumerate(pool.imap_unordered(_eval_combo_deep, top), 1):
            deep_results.append(result)
            pd, avg, per_seed = result
            print(f"  [{i}/{top_n}] avg={avg:.1f}  per_seed={[f'{s:.1f}' for s in per_seed]}")

    deep_results.sort(key=lambda x: -x[1])
    best_params, best_score, best_per_seed = deep_results[0]

    print(f"\nR{rn} BEST: {best_score:.1f}")
    for k in sorted(best_params.keys()):
        print(f"  {k}={best_params[k]}")

    return {
        "round": rn,
        "settle_rate": settle_rate,
        "score": best_score,
        "params": best_params,
        "per_seed": best_per_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, nargs="*", default=[7, 12])
    parser.add_argument("--mc-deep", type=int, default=500)
    args = parser.parse_args()

    n_cpus = mp.cpu_count()
    print(f"CPUs: {n_cpus}")

    results = {}
    for rn in args.rounds:
        result = tune_round(rn, n_cpus, args.mc_deep)
        if result:
            results[rn] = result

    # Save
    out_path = os.path.join(CACHE_DIR, "r7_r12_deep_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for rn, r in results.items():
        print(f"R{rn}: {r['score']:.1f} (settle_rate={r['settle_rate']:.3f})")


if __name__ == "__main__":
    main()
