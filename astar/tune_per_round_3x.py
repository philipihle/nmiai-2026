"""Tune per-round with 3x expanded parameter grid.

Same as tune_per_round.py but with finer granularity and wider ranges.
~3x more combos per round for deeper search.
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
        probs = sim.monte_carlo(n_runs=30, n_steps=50, n_workers=1)
        score = score_prediction(probs, sd["gt"])
        scores.append(score)
    return (params_dict, float(np.mean(scores)), scores)


def _eval_combo_deep(params_dict):
    params = SimParams(**params_dict)
    scores = []
    for sd in _worker_seeds_data:
        sim = AstarSimulator(sd["grid"], sd["settlements"], params=params, seed=456)
        probs = sim.monte_carlo(n_runs=500, n_steps=50, n_workers=1)
        score = score_prediction(probs, sd["gt"])
        scores.append(score)
    return (params_dict, float(np.mean(scores)), scores)


def build_combos_3x(settle_rate):
    """3x expanded search space with finer granularity."""
    fixed = dict(
        spawn_max_per_step=40,
        port_wealth_threshold=0.0,
        wealth_decay=0.0,
        wealth_coastal_rate=0.01,
    )

    if settle_rate < 0.05:
        grid = {
            "spawn_prob":        [0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18],
            "spawn_pop_threshold": [0.2, 0.3, 0.5, 0.7],
            "death_base_rate":   [0.03, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
            "death_food_factor": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
            "pop_growth_rate":   [0.04, 0.06, 0.08, 0.10, 0.12],
            "food_base_regen":   [0.08, 0.10, 0.12, 0.15, 0.20, 0.25],
            "food_competition":  [0.01, 0.02, 0.03, 0.05, 0.07],
            "food_pop_drain":    [0.03, 0.04, 0.06, 0.08, 0.10],
            "food_forest_bonus": [0.01, 0.02, 0.04, 0.06],
            "port_survival_bonus": [0.0, 0.02, 0.03, 0.05],
            "port_prob":         [0.05, 0.08, 0.10, 0.15],
        }
    elif settle_rate < 0.12:
        grid = {
            "spawn_prob":        [0.04, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12],
            "spawn_pop_threshold": [0.2, 0.3, 0.4, 0.5, 0.7],
            "death_base_rate":   [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04],
            "death_food_factor": [0.01, 0.02, 0.03, 0.04, 0.06, 0.08],
            "pop_growth_rate":   [0.06, 0.08, 0.10, 0.12, 0.14],
            "food_base_regen":   [0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
            "food_competition":  [0.01, 0.02, 0.03, 0.04, 0.05],
            "food_pop_drain":    [0.03, 0.04, 0.06, 0.08, 0.10],
            "food_forest_bonus": [0.01, 0.02, 0.04, 0.06, 0.08],
            "port_survival_bonus": [0.0, 0.01, 0.02, 0.03],
            "port_prob":         [0.05, 0.08, 0.10, 0.12, 0.15],
        }
    elif settle_rate < 0.18:
        grid = {
            "spawn_prob":        [0.04, 0.06, 0.07, 0.08, 0.10, 0.12, 0.14],
            "spawn_pop_threshold": [0.2, 0.3, 0.4, 0.5, 0.7],
            "death_base_rate":   [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03],
            "death_food_factor": [0.01, 0.02, 0.03, 0.04, 0.06, 0.08],
            "pop_growth_rate":   [0.06, 0.08, 0.10, 0.12, 0.14],
            "food_base_regen":   [0.15, 0.18, 0.20, 0.25, 0.30, 0.35],
            "food_competition":  [0.01, 0.02, 0.03, 0.04, 0.05],
            "food_pop_drain":    [0.03, 0.04, 0.06, 0.08, 0.10],
            "food_forest_bonus": [0.01, 0.02, 0.04, 0.06, 0.08],
            "port_survival_bonus": [0.0, 0.01, 0.02, 0.03],
            "port_prob":         [0.05, 0.08, 0.10, 0.12, 0.15],
        }
    elif settle_rate < 0.30:
        grid = {
            "spawn_prob":        [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
            "spawn_pop_threshold": [0.2, 0.3, 0.4, 0.5, 0.7],
            "death_base_rate":   [0.003, 0.005, 0.008, 0.01, 0.015, 0.02],
            "death_food_factor": [0.01, 0.02, 0.03, 0.04, 0.06],
            "pop_growth_rate":   [0.08, 0.10, 0.12, 0.14, 0.16],
            "food_base_regen":   [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
            "food_competition":  [0.005, 0.01, 0.02, 0.03, 0.04],
            "food_pop_drain":    [0.03, 0.04, 0.06, 0.08, 0.10],
            "food_forest_bonus": [0.02, 0.04, 0.06, 0.08, 0.10],
            "port_survival_bonus": [0.0, 0.01, 0.02],
            "port_prob":         [0.05, 0.08, 0.10, 0.12, 0.15],
        }
    else:
        # Very high settle rate (R18 territory: 0.48+)
        grid = {
            "spawn_prob":        [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
            "spawn_pop_threshold": [0.2, 0.3, 0.4, 0.5, 0.7],
            "death_base_rate":   [0.002, 0.003, 0.005, 0.008, 0.01, 0.015],
            "death_food_factor": [0.01, 0.02, 0.03, 0.04, 0.06],
            "pop_growth_rate":   [0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
            "food_base_regen":   [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
            "food_competition":  [0.005, 0.01, 0.015, 0.02, 0.03],
            "food_pop_drain":    [0.02, 0.04, 0.06, 0.08, 0.10],
            "food_forest_bonus": [0.04, 0.06, 0.08, 0.10, 0.12],
            "port_survival_bonus": [0.0, 0.01, 0.02],
            "port_prob":         [0.05, 0.08, 0.10, 0.15, 0.20],
        }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    total_full = 1
    for v in values:
        total_full *= len(v)

    max_combos = 150000
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
            vals = tuple(float(rng.choice(grid[k])) for k in keys)
            key = tuple(round(v, 4) for v in vals)
            if key not in seen:
                seen.add(key)
                d = dict(zip(keys, [float(v) for v in vals]))
                d.update(fixed)
                combos.append(d)

    return combos, keys


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


def tune_one_round(rn, n_cpus):
    seeds, settle_rate = load_round_seeds(rn)
    if not seeds:
        print(f"R{rn}: No data, skipping")
        return None

    print(f"\n{'='*60}")
    print(f"R{rn}: settle_rate={settle_rate:.3f}, {len(seeds)} seeds")
    print(f"{'='*60}")

    combos, keys = build_combos_3x(settle_rate)
    total = len(combos)
    print(f"{total} combos x {len(seeds)} seeds x 30 MC (coarse)")

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

    # Deep eval top 10 with 500 MC
    top_n = min(10, len(results))
    top = [r[0] for r in results[:top_n]]
    print(f"\nDeep eval top {top_n} (500 MC)...")
    deep_results = []
    with mp.Pool(n_cpus, initializer=_init_worker, initargs=(seeds,)) as pool:
        for i, result in enumerate(pool.imap_unordered(_eval_combo_deep, top), 1):
            deep_results.append(result)
            pd, avg, per_seed = result
            print(f"  [{i}/{top_n}] avg={avg:.1f}  seeds={[f'{s:.1f}' for s in per_seed]}")

    deep_results.sort(key=lambda x: -x[1])
    best_params, best_score, _ = deep_results[0]

    print(f"\nR{rn} BEST: {best_score:.1f}")
    for k in sorted(best_params.keys()):
        if k not in ("spawn_max_per_step", "port_wealth_threshold", "wealth_decay", "wealth_coastal_rate"):
            print(f"  {k}={best_params[k]}")

    return {
        "round": rn,
        "settle_rate": settle_rate,
        "score": best_score,
        "params": best_params,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    n_cpus = mp.cpu_count()
    print(f"CPUs: {n_cpus}")

    results_path = os.path.join(CACHE_DIR, "per_round_params.json")

    available_rounds = []
    for rn in range(1, 30):
        if os.path.exists(os.path.join(CACHE_DIR, f"round{rn}_detail.json")):
            available_rounds.append(rn)

    target_rounds = args.rounds or available_rounds
    print(f"Tuning rounds (3x grid): {target_rounds}")

    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
        existing_map = {r["round"]: r for r in existing}
    else:
        existing_map = {}

    for rn in target_rounds:
        result = tune_one_round(rn, n_cpus)
        if result:
            # Only update if better than existing
            old = existing_map.get(rn, {}).get("score", 0)
            if result["score"] > old:
                existing_map[rn] = result
                print(f"R{rn}: NEW BEST {result['score']:.1f} (was {old:.1f})")
            else:
                print(f"R{rn}: keeping old {old:.1f} > new {result['score']:.1f}")
            with open(results_path, "w") as f:
                json.dump(list(existing_map.values()), f, indent=2)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
