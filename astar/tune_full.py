"""Comprehensive simulator parameter tuning with all discovered mechanics.

Includes: d=3 spawn, multi-spawn, port_survival_bonus, food params, food_forest_bonus.
Uses dict-based combos for clarity.

Usage:
  python tune_full.py --round-type normal
  python tune_full.py --round-type dieoff
  python tune_full.py --round-type normal --partition 0 --num-partitions 2  # split across VMs
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
NORMAL_ROUNDS = {1, 2, 4, 5, 6, 7}
DIEOFF_ROUNDS = {3, 8}

_worker_seeds_data = None


def _init_worker(seeds_data):
    global _worker_seeds_data
    _worker_seeds_data = seeds_data


def _eval_combo_coarse(params_dict):
    """Evaluate one param dict across all cached seeds with 20 MC runs."""
    params = SimParams(**params_dict)
    scores = []
    for sd in _worker_seeds_data:
        sim = AstarSimulator(sd["grid"], sd["settlements"], params=params, seed=123)
        probs = sim.monte_carlo(n_runs=20, n_steps=50, n_workers=1)
        score = score_prediction(probs, sd["gt"])
        scores.append(score)
    return (params_dict, float(np.mean(scores)), scores)


def _eval_combo_deep(params_dict):
    """Re-evaluate with 200 MC runs for more accurate scoring."""
    params = SimParams(**params_dict)
    scores = []
    for sd in _worker_seeds_data:
        sim = AstarSimulator(sd["grid"], sd["settlements"], params=params, seed=456)
        probs = sim.monte_carlo(n_runs=200, n_steps=50, n_workers=1)
        score = score_prediction(probs, sd["gt"])
        scores.append(score)
    return (params_dict, float(np.mean(scores)), scores)


def build_combos_normal():
    """Full search space for normal rounds incorporating all replay findings.
    Anchored around food-tune best (sp=0.10, th=0.5, db=0.01, df=0.04,
    pgr=0.10, fbr=0.25, fc=0.03, fd=0.08) but with new params.
    ~15K combos — manageable on 192 CPUs in ~15 min.
    """
    grid = {
        "spawn_prob":        [0.08, 0.10, 0.12],
        "spawn_pop_threshold": [0.3, 0.5],
        "death_base_rate":   [0.005, 0.01, 0.02],
        "death_food_factor": [0.02, 0.04, 0.06],
        "pop_growth_rate":   [0.08, 0.10, 0.12],
        "food_base_regen":   [0.20, 0.25, 0.30],
        "food_competition":  [0.01, 0.02, 0.03],
        "food_pop_drain":    [0.04, 0.06, 0.08],
        "food_forest_bonus": [0.02, 0.04, 0.06],
        "port_survival_bonus": [0.0, 0.02, 0.04],
    }
    # Fixed params
    fixed = dict(
        spawn_max_per_step=40,
        port_prob=0.10,
        port_wealth_threshold=0.0,
        wealth_decay=0.0,
        wealth_coastal_rate=0.01,
    )

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        d = dict(zip(keys, vals))
        d.update(fixed)
        combos.append(d)
    return combos, keys


def build_combos_dieoff():
    """Full search space for die-off rounds."""
    grid = {
        "spawn_prob":        [0.05, 0.08, 0.10, 0.15],
        "spawn_pop_threshold": [0.5, 0.7],
        "death_base_rate":   [0.05, 0.08, 0.10, 0.15],
        "death_food_factor": [0.05, 0.10, 0.15],
        "pop_growth_rate":   [0.06, 0.08, 0.10],
        "food_base_regen":   [0.10, 0.15, 0.20],
        "food_competition":  [0.01, 0.03, 0.05],
        "food_pop_drain":    [0.04, 0.06, 0.08],
        "food_forest_bonus": [0.02, 0.04],
        "port_survival_bonus": [0.0, 0.03],
    }
    fixed = dict(
        spawn_max_per_step=40,
        port_prob=0.10,
        port_wealth_threshold=0.0,
        wealth_decay=0.0,
        wealth_coastal_rate=0.01,
    )

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = []
    for vals in itertools.product(*values):
        d = dict(zip(keys, vals))
        d.update(fixed)
        combos.append(d)
    return combos, keys


def load_all_seeds():
    """Load all cached seeds."""
    seeds = []
    for rn in range(1, 20):
        detail_path = os.path.join(CACHE_DIR, f"round{rn}_detail.json")
        if not os.path.exists(detail_path):
            continue
        with open(detail_path) as f:
            detail = json.load(f)
        for si in range(detail.get("seeds_count", 5)):
            replay_path = os.path.join(CACHE_DIR, f"round{rn}_seed{si}_replay.json")
            analysis_path = os.path.join(CACHE_DIR, f"round{rn}_seed{si}_analysis.json")
            if not os.path.exists(replay_path) or not os.path.exists(analysis_path):
                continue
            with open(replay_path) as f:
                replay = json.load(f)
            with open(analysis_path) as f:
                analysis = json.load(f)
            seeds.append({
                "round": rn,
                "seed": si,
                "grid": np.array(detail["initial_states"][si]["grid"]),
                "settlements": replay["frames"][0]["settlements"],
                "gt": np.array(analysis["ground_truth"]),
            })
    return seeds


def fmt_params(d, keys):
    """Format params dict for display."""
    return " ".join(f"{k}={d[k]:.3f}" if isinstance(d[k], float) else f"{k}={d[k]}" for k in keys)


def main():
    parser = argparse.ArgumentParser(description="Full simulator parameter tuning")
    parser.add_argument("--round-type", choices=["normal", "dieoff"], required=True)
    parser.add_argument("--partition", type=int, default=0, help="Which partition to run (0-based)")
    parser.add_argument("--num-partitions", type=int, default=1, help="Total number of partitions")
    args = parser.parse_args()

    n_cpus = mp.cpu_count()
    print(f"CPUs: {n_cpus}")
    print(f"Round type: {args.round_type}")

    all_seeds = load_all_seeds()
    target_rounds = NORMAL_ROUNDS if args.round_type == "normal" else DIEOFF_ROUNDS
    filtered_seeds = [s for s in all_seeds if s["round"] in target_rounds]
    print(f"Loaded {len(filtered_seeds)} seeds from rounds {sorted(target_rounds)}")

    if not filtered_seeds:
        print("No seeds found!")
        return

    for rn in sorted(set(s["round"] for s in filtered_seeds)):
        count = sum(1 for s in filtered_seeds if s["round"] == rn)
        print(f"  Round {rn}: {count} seeds")

    if args.round_type == "normal":
        combos, keys = build_combos_normal()
    else:
        combos, keys = build_combos_dieoff()

    total_all = len(combos)

    # Partition if running across multiple VMs
    if args.num_partitions > 1:
        chunk_size = len(combos) // args.num_partitions
        start = args.partition * chunk_size
        end = start + chunk_size if args.partition < args.num_partitions - 1 else len(combos)
        combos = combos[start:end]
        print(f"Partition {args.partition}/{args.num_partitions}: combos {start}-{end} of {total_all}")

    total = len(combos)
    print(f"\n{total} param combos x {len(filtered_seeds)} seeds x 20 MC runs (coarse)")
    print(f"Parallelizing across {n_cpus} CPUs...", flush=True)

    # Phase 1: Coarse
    t0 = time.time()
    results = []
    with mp.Pool(n_cpus, initializer=_init_worker, initargs=(filtered_seeds,)) as pool:
        for i, result in enumerate(pool.imap_unordered(_eval_combo_coarse, combos), 1):
            results.append(result)
            if i % (n_cpus * 2) == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                best_so_far = max(r[1] for r in results)
                print(f"  [{i:>6d}/{total}] {i*100/total:.0f}%  {elapsed:.0f}s  ETA {eta:.0f}s  best={best_so_far:.1f}", flush=True)

    elapsed = time.time() - t0
    results.sort(key=lambda x: -x[1])
    print(f"\nCoarse done in {elapsed:.0f}s")

    # Top 20
    print(f"\nTop 20 coarse:")
    print(f"{'rank':>4} | {'avg':>5} | params")
    print("-" * 120)
    for rank, (pd, avg, scores) in enumerate(results[:20]):
        round_avgs = {}
        for sd, sc in zip(filtered_seeds, scores):
            rn = sd["round"]
            if rn not in round_avgs:
                round_avgs[rn] = []
            round_avgs[rn].append(sc)
        per_round = " ".join(f"R{rn}={np.mean(scs):.0f}" for rn, scs in sorted(round_avgs.items()))
        short = " ".join(f"{k[:6]}={pd[k]}" for k in keys)
        print(f"{rank+1:4d} | {avg:5.1f} | {short} | {per_round}")

    # Phase 2: Deep eval top 10
    top10 = [r[0] for r in results[:10]]
    print(f"\n=== Deep eval top 10 with 200 MC runs ===")

    t1 = time.time()
    deep_results = []
    with mp.Pool(n_cpus, initializer=_init_worker, initargs=(filtered_seeds,)) as pool:
        for i, result in enumerate(pool.imap_unordered(_eval_combo_deep, top10), 1):
            deep_results.append(result)
            pd, avg, scores = result
            print(f"  [{i}/10] avg={avg:.1f}", flush=True)

    deep_results.sort(key=lambda x: -x[1])
    elapsed2 = time.time() - t1
    print(f"\nDeep eval done in {elapsed2:.0f}s")

    print(f"\n=== Final ranking ({args.round_type}) ===")
    print(f"{'rank':>4} | {'avg':>5} | params | per-round")
    print("-" * 140)
    for rank, (pd, avg, scores) in enumerate(deep_results):
        round_avgs = {}
        for sd, sc in zip(filtered_seeds, scores):
            rn = sd["round"]
            if rn not in round_avgs:
                round_avgs[rn] = []
            round_avgs[rn].append(sc)
        per_round = " ".join(f"R{rn}={np.mean(scs):.0f}" for rn, scs in sorted(round_avgs.items()))
        short = " ".join(f"{k[:6]}={pd[k]}" for k in keys)
        print(f"{rank+1:4d} | {avg:5.1f} | {short} | {per_round}")

    # Best params as code
    best = deep_results[0][0]
    print(f"\n=== Best {args.round_type.upper()} params ===")
    print("SimParams(")
    for k, v in sorted(best.items()):
        print(f"    {k}={v!r},")
    print(")")


if __name__ == "__main__":
    main()
