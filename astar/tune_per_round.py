"""Tune optimal SimParams per round, then build interpolation function.

Phase 1: For each round, grid-search optimal MC-only params
Phase 2: Fit linear interpolation: settle_rate → optimal param values
Phase 3: Output params_from_rate() function for solver.py

Usage:
  python tune_per_round.py                    # tune all rounds
  python tune_per_round.py --rounds 1 2 6     # tune specific rounds
  python tune_per_round.py --fit-only          # skip tuning, just fit from cached results
"""
import argparse
import json
import os
import numpy as np
import time
import itertools
import multiprocessing as mp
from simulator import AstarSimulator, SimParams, _grid_to_counts, TERRAIN_TO_CLASS
from backtest import score_prediction

CACHE_DIR = "replay_cache"

_worker_seeds_data = None


def _init_worker(seeds_data):
    global _worker_seeds_data
    _worker_seeds_data = seeds_data


def _eval_combo(params_dict):
    """Evaluate one param dict across seeds for ONE round."""
    params = SimParams(**params_dict)
    scores = []
    for sd in _worker_seeds_data:
        sim = AstarSimulator(sd["grid"], sd["settlements"], params=params, seed=123)
        probs = sim.monte_carlo(n_runs=20, n_steps=50, n_workers=1)
        score = score_prediction(probs, sd["gt"])
        scores.append(score)
    return (params_dict, float(np.mean(scores)), scores)


def _eval_combo_deep(params_dict):
    """Re-evaluate with 200 MC runs."""
    params = SimParams(**params_dict)
    scores = []
    for sd in _worker_seeds_data:
        sim = AstarSimulator(sd["grid"], sd["settlements"], params=params, seed=456)
        probs = sim.monte_carlo(n_runs=200, n_steps=50, n_workers=1)
        score = score_prediction(probs, sd["gt"])
        scores.append(score)
    return (params_dict, float(np.mean(scores)), scores)


def build_combos(settle_rate):
    """Build search space adapted to settle rate.
    Higher settle rate → search higher spawn/growth values.
    Lower settle rate → search higher death values.
    """
    fixed = dict(
        spawn_max_per_step=40,
        port_prob=0.10,
        port_wealth_threshold=0.0,
        wealth_decay=0.0,
        wealth_coastal_rate=0.01,
    )

    if settle_rate < 0.05:
        # Die-off
        grid = {
            "spawn_prob":        [0.05, 0.08, 0.10, 0.15],
            "spawn_pop_threshold": [0.3, 0.5, 0.7],
            "death_base_rate":   [0.05, 0.08, 0.10, 0.15, 0.20],
            "death_food_factor": [0.05, 0.10, 0.15],
            "pop_growth_rate":   [0.06, 0.08, 0.10],
            "food_base_regen":   [0.10, 0.15, 0.20],
            "food_competition":  [0.01, 0.03, 0.05],
            "food_pop_drain":    [0.04, 0.06, 0.08],
            "food_forest_bonus": [0.02, 0.04],
            "port_survival_bonus": [0.0, 0.03, 0.05],
        }
    elif settle_rate < 0.12:
        # Low normal
        grid = {
            "spawn_prob":        [0.06, 0.08, 0.10],
            "spawn_pop_threshold": [0.3, 0.5],
            "death_base_rate":   [0.005, 0.01, 0.02, 0.03],
            "death_food_factor": [0.02, 0.04, 0.06],
            "pop_growth_rate":   [0.08, 0.10, 0.12],
            "food_base_regen":   [0.15, 0.20, 0.25],
            "food_competition":  [0.02, 0.03, 0.04],
            "food_pop_drain":    [0.04, 0.06, 0.08],
            "food_forest_bonus": [0.02, 0.04, 0.06],
            "port_survival_bonus": [0.0, 0.02],
        }
    elif settle_rate < 0.18:
        # Medium normal
        grid = {
            "spawn_prob":        [0.06, 0.08, 0.10, 0.12],
            "spawn_pop_threshold": [0.3, 0.5],
            "death_base_rate":   [0.005, 0.01, 0.02],
            "death_food_factor": [0.02, 0.04, 0.06],
            "pop_growth_rate":   [0.08, 0.10, 0.12],
            "food_base_regen":   [0.20, 0.25, 0.30],
            "food_competition":  [0.01, 0.02, 0.03],
            "food_pop_drain":    [0.04, 0.06, 0.08],
            "food_forest_bonus": [0.02, 0.04, 0.06],
            "port_survival_bonus": [0.0, 0.02],
        }
    else:
        # High normal
        grid = {
            "spawn_prob":        [0.06, 0.08, 0.10, 0.12],
            "spawn_pop_threshold": [0.3, 0.5],
            "death_base_rate":   [0.005, 0.01, 0.015],
            "death_food_factor": [0.02, 0.04, 0.06],
            "pop_growth_rate":   [0.10, 0.12, 0.15],
            "food_base_regen":   [0.20, 0.25, 0.30, 0.35],
            "food_competition":  [0.01, 0.02, 0.03],
            "food_pop_drain":    [0.04, 0.06, 0.08],
            "food_forest_bonus": [0.04, 0.06, 0.08],
            "port_survival_bonus": [0.0, 0.02],
        }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        d = dict(zip(keys, vals))
        d.update(fixed)
        combos.append(d)
    return combos, keys


def load_round_seeds(rn):
    """Load all seeds for one round."""
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
            "round": rn,
            "seed": si,
            "grid": np.array(detail["initial_states"][si]["grid"]),
            "settlements": replay["frames"][0]["settlements"],
            "gt": gt,
        })
    return seeds, np.mean(rates) if rates else 0.0


def tune_one_round(rn, n_cpus):
    """Grid-search optimal params for one round."""
    seeds, settle_rate = load_round_seeds(rn)
    if not seeds:
        print(f"R{rn}: No data, skipping")
        return None

    print(f"\n{'='*60}")
    print(f"R{rn}: settle_rate={settle_rate:.3f}, {len(seeds)} seeds")
    print(f"{'='*60}")

    combos, keys = build_combos(settle_rate)
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
                print(f"  [{i:>5d}/{total}] {i*100/total:.0f}%  {elapsed:.0f}s  ETA {eta:.0f}s  best={best:.1f}", flush=True)

    results.sort(key=lambda x: -x[1])

    # Deep eval top 5
    top5 = [r[0] for r in results[:5]]
    print(f"\nDeep eval top 5 (200 MC)...")
    deep_results = []
    with mp.Pool(n_cpus, initializer=_init_worker, initargs=(seeds,)) as pool:
        for i, result in enumerate(pool.imap_unordered(_eval_combo_deep, top5), 1):
            deep_results.append(result)
            pd, avg, _ = result
            print(f"  [{i}/5] avg={avg:.1f}")

    deep_results.sort(key=lambda x: -x[1])
    best_params, best_score, _ = deep_results[0]

    print(f"\nR{rn} BEST: {best_score:.1f}")
    for k in keys:
        if k in best_params:
            print(f"  {k}={best_params[k]}")

    return {
        "round": rn,
        "settle_rate": settle_rate,
        "score": best_score,
        "params": best_params,
    }


def fit_interpolation(results):
    """Fit linear interpolation from settle_rate to each param."""
    print(f"\n{'='*60}")
    print("FITTING INTERPOLATION")
    print(f"{'='*60}")

    # Sort by settle rate
    results.sort(key=lambda r: r["settle_rate"])

    # Extract data
    rates = [r["settle_rate"] for r in results]
    param_keys = [k for k in results[0]["params"].keys()
                  if k not in ("spawn_max_per_step", "port_prob", "port_wealth_threshold",
                               "wealth_decay", "wealth_coastal_rate")]

    print(f"\nData points: {len(results)}")
    print(f"Rate range: {min(rates):.3f} - {max(rates):.3f}")
    print(f"\nRound | Rate  | Score | Key params")
    print("-" * 80)
    for r in results:
        p = r["params"]
        print(f"R{r['round']:>2}   | {r['settle_rate']:.3f} | {r['score']:5.1f} | "
              f"sp={p['spawn_prob']:.2f} db={p['death_base_rate']:.3f} "
              f"fbr={p['food_base_regen']:.2f} fc={p['food_competition']:.2f} "
              f"ffb={p['food_forest_bonus']:.2f}")

    # Fit each param
    print(f"\nInterpolation coefficients (param = a * settle_rate + b):")
    print(f"{'param':<25} {'a':>8} {'b':>8} {'R²':>6}")
    print("-" * 50)

    coeffs = {}
    for key in param_keys:
        vals = [r["params"].get(key, 0) for r in results]
        if len(set(vals)) <= 1:
            coeffs[key] = {"a": 0, "b": vals[0], "r2": 1.0}
            continue
        # Linear fit
        a, b = np.polyfit(rates, vals, 1)
        # R²
        pred = [a * r + b for r in rates]
        ss_res = sum((v - p) ** 2 for v, p in zip(vals, pred))
        ss_tot = sum((v - np.mean(vals)) ** 2 for v in vals)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        coeffs[key] = {"a": float(a), "b": float(b), "r2": float(r2)}
        print(f"{key:<25} {a:>8.4f} {b:>8.4f} {r2:>6.3f}")

    # Generate code
    print(f"\n{'='*60}")
    print("GENERATED CODE FOR solver.py")
    print(f"{'='*60}")

    # Store raw calibration points for nearest-neighbor fallback
    print(f"""
# --- Per-round parameter interpolation ---
# Generated by tune_per_round.py on {time.strftime('%Y-%m-%d %H:%M')}
# Calibration points: {len(results)} rounds

_CALIBRATION_POINTS = [""")
    for r in results:
        print(f"    {{'rate': {r['settle_rate']:.4f}, 'round': {r['round']}, 'score': {r['score']:.1f}, 'params': {{")
        for k in param_keys:
            v = r['params'].get(k, 0)
            print(f"        '{k}': {v},")
        print(f"    }}}},")
    print("""]

def params_from_rate(settle_rate):
    \"\"\"Interpolate SimParams from observed settle rate.
    Uses weighted blend of two nearest calibration points.\"\"\"
    from simulator import SimParams
    pts = _CALIBRATION_POINTS
    rates = [p['rate'] for p in pts]

    # Clamp to range
    if settle_rate <= rates[0]:
        return SimParams(**pts[0]['params'])
    if settle_rate >= rates[-1]:
        return SimParams(**pts[-1]['params'])

    # Find bracketing points
    for i in range(len(rates) - 1):
        if rates[i] <= settle_rate <= rates[i+1]:
            lo, hi = pts[i], pts[i+1]
            t = (settle_rate - lo['rate']) / (hi['rate'] - lo['rate'])
            blended = {}
            for k in lo['params']:
                blended[k] = lo['params'][k] * (1 - t) + hi['params'][k] * t
            return SimParams(**blended)

    return SimParams(**pts[len(pts)//2]['params'])
""")

    return coeffs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, nargs="*", default=None)
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()

    n_cpus = mp.cpu_count()
    print(f"CPUs: {n_cpus}")

    results_path = os.path.join(CACHE_DIR, "per_round_params.json")

    if args.fit_only:
        with open(results_path) as f:
            results = json.load(f)
        fit_interpolation(results)
        return

    available_rounds = []
    for rn in range(1, 20):
        if os.path.exists(os.path.join(CACHE_DIR, f"round{rn}_detail.json")):
            available_rounds.append(rn)

    target_rounds = args.rounds or available_rounds
    print(f"Tuning rounds: {target_rounds}")

    # Load existing results
    if os.path.exists(results_path):
        with open(results_path) as f:
            existing = json.load(f)
        existing_map = {r["round"]: r for r in existing}
    else:
        existing_map = {}

    for rn in target_rounds:
        result = tune_one_round(rn, n_cpus)
        if result:
            existing_map[rn] = result
            # Save after each round
            with open(results_path, "w") as f:
                json.dump(list(existing_map.values()), f, indent=2)
            print(f"Saved to {results_path}")

    results = list(existing_map.values())
    if len(results) >= 3:
        fit_interpolation(results)


if __name__ == "__main__":
    main()
