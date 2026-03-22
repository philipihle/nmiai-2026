"""Tune simulator parameters, split by round type (normal vs dieoff).

Usage:
  python tune_sim.py --round-type normal              # tunes on R1,R2,R4,R5,R6,R7
  python tune_sim.py --round-type dieoff              # tunes on R3,R8
  python tune_sim.py --round-type normal --with-raiding   # include raiding params
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

# Round classification
NORMAL_ROUNDS = {1, 2, 4, 5, 6, 7}
DIEOFF_ROUNDS = {3, 8}

# Worker globals
_worker_seeds_data = None


def _init_multi_round_worker(seeds_data):
    global _worker_seeds_data
    _worker_seeds_data = seeds_data


def _combo_to_params(combo):
    """Convert a combo tuple to SimParams."""
    base = dict(spawn_max_per_step=40, port_prob=0.10, wealth_decay=0.0, wealth_coastal_rate=0.01)
    if len(combo) == 8 and isinstance(combo[-1], int):
        # Raiding: (sp, th, db, df, pgr, fbr, raid_bp, raid_range)
        sp, th, db, df, pgr, fbr, raid_bp, raid_rng = combo
        return SimParams(spawn_prob=sp, spawn_pop_threshold=th,
                         death_base_rate=db, death_food_factor=df,
                         pop_growth_rate=pgr, food_base_regen=fbr,
                         raid_base_prob=raid_bp, raid_range=raid_rng, **base)
    elif len(combo) == 8 and all(isinstance(x, float) for x in combo):
        # Food-extended: (sp, th, db, df, pgr, fbr, f_comp, f_drain)
        sp, th, db, df, pgr, fbr, f_comp, f_drain = combo
        return SimParams(spawn_prob=sp, spawn_pop_threshold=th,
                         death_base_rate=db, death_food_factor=df,
                         pop_growth_rate=pgr, food_base_regen=fbr,
                         food_competition=f_comp, food_pop_drain=f_drain, **base)
    elif len(combo) == 8:
        # Death wave: (sp, th, db, df, pgr, fbr, dw_amp, dw_period)
        sp, th, db, df, pgr, fbr, dw_amp, dw_per = combo
        return SimParams(spawn_prob=sp, spawn_pop_threshold=th,
                         death_base_rate=db, death_food_factor=df,
                         pop_growth_rate=pgr, food_base_regen=fbr,
                         death_wave_amplitude=dw_amp, death_wave_period=dw_per, **base)
    else:
        sp, th, db, df, pgr, fbr = combo
        return SimParams(spawn_prob=sp, spawn_pop_threshold=th,
                         death_base_rate=db, death_food_factor=df,
                         pop_growth_rate=pgr, food_base_regen=fbr, **base)


def _eval_combo_all_rounds(combo):
    """Evaluate one param combo across all cached seeds. Returns avg score."""
    params = _combo_to_params(combo)
    scores = []
    for sd in _worker_seeds_data:
        sim = AstarSimulator(sd["grid"], sd["settlements"], params=params, seed=123)
        probs = sim.monte_carlo(n_runs=20, n_steps=50, n_workers=1)
        score = score_prediction(probs, sd["gt"])
        scores.append(score)
    return (combo, float(np.mean(scores)), scores)


def _eval_combo_deep(combo):
    """Re-evaluate with 200 MC runs for more accurate scoring."""
    params = _combo_to_params(combo)
    scores = []
    for sd in _worker_seeds_data:
        sim = AstarSimulator(sd["grid"], sd["settlements"], params=params, seed=456)
        probs = sim.monte_carlo(n_runs=200, n_steps=50, n_workers=1)
        score = score_prediction(probs, sd["gt"])
        scores.append(score)
    return (combo, float(np.mean(scores)), scores)


def build_combos_with_raiding(round_type):
    """Build parameter combinations including raiding params."""
    if round_type == "normal":
        return list(itertools.product(
            [0.15, 0.20, 0.25],                 # spawn_prob
            [0.5],                                # spawn_threshold
            [0.03, 0.04, 0.05],                   # death_base
            [0.02, 0.04],                          # death_food
            [0.08, 0.10],                          # pop_growth_rate
            [0.10, 0.15],                          # food_base_regen
            [0.0, 0.03, 0.06, 0.10],              # raid_base_prob
            [3, 5],                                # raid_range
        ))
    else:  # dieoff
        return list(itertools.product(
            [0.10, 0.15],                          # spawn_prob
            [0.5],                                 # spawn_threshold
            [0.10, 0.15],                           # death_base
            [0.05, 0.10],                           # death_food
            [0.06, 0.08],                           # pop_growth_rate
            [0.10, 0.15],                           # food_base_regen
            [0.0, 0.03, 0.06],                     # raid_base_prob
            [3, 5],                                 # raid_range
        ))


def build_combos_with_waves(round_type):
    """Build parameter combinations including death wave params."""
    if round_type == "normal":
        return list(itertools.product(
            [0.15, 0.20, 0.25],                 # spawn_prob
            [0.5],                                # spawn_threshold
            [0.03, 0.04, 0.05],                   # death_base
            [0.02, 0.04],                          # death_food
            [0.08, 0.10],                          # pop_growth_rate
            [0.10, 0.15],                          # food_base_regen
            [0.0, 0.3, 0.5, 0.8, 1.0],            # death_wave_amplitude
            [3.0, 4.0, 5.0, 7.0],                 # death_wave_period
        ))
    else:  # dieoff
        return list(itertools.product(
            [0.10, 0.15],                          # spawn_prob
            [0.5],                                 # spawn_threshold
            [0.10, 0.15],                           # death_base
            [0.05, 0.10],                           # death_food
            [0.06, 0.08],                           # pop_growth_rate
            [0.10, 0.15],                           # food_base_regen
            [0.0, 0.3, 0.5, 0.8],                  # death_wave_amplitude
            [3.0, 5.0, 7.0],                       # death_wave_period
        ))


def build_combos_with_food(round_type):
    """Build parameter combinations with extended food params."""
    if round_type == "normal":
        return list(itertools.product(
            [0.08, 0.10, 0.12, 0.15],             # spawn_prob
            [0.3, 0.5],                             # spawn_threshold
            [0.01, 0.02, 0.03, 0.04],              # death_base
            [0.02, 0.04],                            # death_food
            [0.08, 0.10, 0.12],                      # pop_growth
            [0.15, 0.20, 0.25, 0.30],               # food_base_regen (extended up)
            [0.00, 0.01, 0.02, 0.03],               # food_competition (new)
            [0.04, 0.06, 0.08],                      # food_pop_drain (new)
        ))
    else:  # dieoff
        return list(itertools.product(
            [0.10, 0.15],                            # spawn_prob
            [0.5],                                    # spawn_threshold
            [0.05, 0.10, 0.15],                       # death_base
            [0.05, 0.10],                              # death_food
            [0.06, 0.08],                              # pop_growth
            [0.10, 0.15, 0.20],                        # food_base_regen
            [0.01, 0.03, 0.05],                        # food_competition
            [0.04, 0.08],                              # food_pop_drain
        ))


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


def build_combos(round_type):
    """Build parameter combinations based on round type."""
    if round_type == "normal":
        return list(itertools.product(
            [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],   # spawn_prob (extended)
            [0.3, 0.4, 0.5, 0.6, 0.7],                # spawn_threshold (extended down)
            [0.02, 0.03, 0.04, 0.05, 0.06],            # death_base
            [0.02, 0.04, 0.06],                         # death_food
            [0.08, 0.10, 0.12, 0.15],                   # pop_growth_rate (extended up)
            [0.10, 0.15, 0.20],                         # food_base_regen
        ))
    else:  # dieoff
        return list(itertools.product(
            [0.02, 0.05, 0.08, 0.10, 0.15],           # spawn_prob
            [0.5, 0.7, 0.9],                           # spawn_threshold
            [0.03, 0.05, 0.07, 0.10, 0.15],            # death_base
            [0.05, 0.10, 0.15, 0.20],                  # death_food
            [0.04, 0.06, 0.08],                         # pop_growth_rate
            [0.05, 0.10, 0.15],                         # food_base_regen
        ))


def main():
    parser = argparse.ArgumentParser(description="Tune SimParams by round type")
    parser.add_argument("--round-type", choices=["normal", "dieoff"], required=True,
                        help="Which round type to tune for")
    parser.add_argument("--with-raiding", action="store_true",
                        help="Include raiding params in search space")
    parser.add_argument("--with-waves", action="store_true",
                        help="Include death wave params in search space")
    parser.add_argument("--with-food", action="store_true",
                        help="Include food_competition and food_pop_drain in search space")
    args = parser.parse_args()

    n_cpus = mp.cpu_count()
    print(f"CPUs: {n_cpus}")
    print(f"Round type: {args.round_type}")

    # Load and filter seeds
    all_seeds = load_all_seeds()
    target_rounds = NORMAL_ROUNDS if args.round_type == "normal" else DIEOFF_ROUNDS
    filtered_seeds = [s for s in all_seeds if s["round"] in target_rounds]
    print(f"Loaded {len(filtered_seeds)} seeds from rounds {sorted(target_rounds)}")

    if not filtered_seeds:
        print("No seeds found for this round type!")
        return

    # Show round distribution
    for rn in sorted(set(s["round"] for s in filtered_seeds)):
        count = sum(1 for s in filtered_seeds if s["round"] == rn)
        print(f"  Round {rn}: {count} seeds")

    # Build param combos
    if args.with_food:
        combos = build_combos_with_food(args.round_type)
    elif args.with_waves:
        combos = build_combos_with_waves(args.round_type)
    elif args.with_raiding:
        combos = build_combos_with_raiding(args.round_type)
    else:
        combos = build_combos(args.round_type)
    total = len(combos)
    print(f"\n{total} param combos x {len(filtered_seeds)} seeds x 20 MC runs (coarse)")
    print(f"Parallelizing across {n_cpus} CPUs (combo-level)...", flush=True)

    # Phase 1: Coarse search with 20 MC runs
    t0 = time.time()
    results = []
    with mp.Pool(n_cpus, initializer=_init_multi_round_worker,
                 initargs=(filtered_seeds,)) as pool:
        for i, result in enumerate(pool.imap_unordered(_eval_combo_all_rounds, combos), 1):
            results.append(result)
            if i % n_cpus == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                best_so_far = max(r[1] for r in results)
                print(f"  [{i:>5d}/{total}] {i*100/total:.0f}%  {elapsed:.0f}s elapsed  ETA {eta:.0f}s  best={best_so_far:.1f}", flush=True)

    elapsed = time.time() - t0
    results.sort(key=lambda x: -x[1])
    print(f"\nCoarse search done in {elapsed:.0f}s")

    # Top 20 from coarse
    has_raiding = args.with_raiding
    has_waves = args.with_waves
    has_food = args.with_food
    has_extra = has_raiding or has_waves or has_food
    print(f"\nTop 20 coarse (avg across {len(filtered_seeds)} seeds):")
    hdr = f"{'rank':>4} | {'avg':>5} | {'sp':>4} {'th':>4} {'db':>5} {'df':>4} {'pgr':>5} {'fbr':>4}"
    if has_food:
        hdr += f" {'fCmp':>5} {'fDrn':>5}"
    elif has_raiding:
        hdr += f" {'rbp':>5} {'rrng':>4}"
    elif has_waves:
        hdr += f" {'dwA':>5} {'dwP':>4}"
    hdr += " | per-round avg"
    print(hdr)
    print("-" * (110 if has_extra else 90))
    for rank, (combo, avg, scores) in enumerate(results[:20]):
        round_avgs = {}
        for sd, sc in zip(filtered_seeds, scores):
            rn = sd["round"]
            if rn not in round_avgs:
                round_avgs[rn] = []
            round_avgs[rn].append(sc)
        per_round = " ".join(f"R{rn}={np.mean(scs):.0f}" for rn, scs in sorted(round_avgs.items()))
        if has_food:
            sp, th, db, df, pgr, fbr, fc, fd = combo
            print(f"{rank+1:4d} | {avg:5.1f} | {sp:.2f} {th:.1f} {db:.3f} {df:.2f} {pgr:.2f}  {fbr:.2f} {fc:.3f} {fd:.3f} | {per_round}")
        elif has_raiding:
            sp, th, db, df, pgr, fbr, rbp, rrng = combo
            print(f"{rank+1:4d} | {avg:5.1f} | {sp:.2f} {th:.1f} {db:.3f} {df:.2f} {pgr:.2f}  {fbr:.2f} {rbp:.3f}  {rrng:>2} | {per_round}")
        elif has_waves:
            sp, th, db, df, pgr, fbr, dwa, dwp = combo
            print(f"{rank+1:4d} | {avg:5.1f} | {sp:.2f} {th:.1f} {db:.3f} {df:.2f} {pgr:.2f}  {fbr:.2f} {dwa:.2f}  {dwp:.1f} | {per_round}")
        else:
            sp, th, db, df, pgr, fbr = combo
            print(f"{rank+1:4d} | {avg:5.1f} | {sp:.2f} {th:.1f} {db:.3f} {df:.2f} {pgr:.2f}  {fbr:.2f} | {per_round}")

    # Phase 2: Re-evaluate top-10 with 200 MC runs
    top10_combos = [r[0] for r in results[:10]]
    print(f"\n=== Re-evaluating top 10 with 200 MC runs per seed ===")

    t1 = time.time()
    deep_results = []
    with mp.Pool(n_cpus, initializer=_init_multi_round_worker,
                 initargs=(filtered_seeds,)) as pool:
        for i, result in enumerate(pool.imap_unordered(_eval_combo_deep, top10_combos), 1):
            deep_results.append(result)
            combo, avg, scores = result
            if has_food and len(combo) == 8:
                sp, th, db, df, pgr, fbr, fc, fd = combo
                print(f"  [{i}/10] avg={avg:.1f}  sp={sp:.2f} th={th:.1f} db={db:.3f} df={df:.2f} pgr={pgr:.2f} fbr={fbr:.2f} fc={fc:.3f} fd={fd:.3f}")
            elif has_raiding and len(combo) == 8:
                sp, th, db, df, pgr, fbr, rbp, rrng = combo
                print(f"  [{i}/10] avg={avg:.1f}  sp={sp:.2f} th={th:.1f} db={db:.3f} df={df:.2f} pgr={pgr:.2f} fbr={fbr:.2f} raid={rbp:.3f} rng={rrng}")
            elif has_waves and len(combo) == 8:
                sp, th, db, df, pgr, fbr, dwa, dwp = combo
                print(f"  [{i}/10] avg={avg:.1f}  sp={sp:.2f} th={th:.1f} db={db:.3f} df={df:.2f} pgr={pgr:.2f} fbr={fbr:.2f} wave={dwa:.2f} per={dwp:.0f}")
            else:
                sp, th, db, df, pgr, fbr = combo
                print(f"  [{i}/10] avg={avg:.1f}  sp={sp:.2f} th={th:.1f} db={db:.3f} df={df:.2f} pgr={pgr:.2f} fbr={fbr:.2f}")

    deep_results.sort(key=lambda x: -x[1])
    elapsed2 = time.time() - t1
    print(f"\nDeep evaluation done in {elapsed2:.0f}s")

    # Final ranking
    print(f"\n=== Final ranking ({args.round_type}) ===")
    hdr2 = f"{'rank':>4} | {'avg':>5} | {'sp':>4} {'th':>4} {'db':>5} {'df':>4} {'pgr':>5} {'fbr':>4}"
    if has_food:
        hdr2 += f" {'fCmp':>5} {'fDrn':>5}"
    elif has_raiding:
        hdr2 += f" {'rbp':>5} {'rrng':>4}"
    elif has_waves:
        hdr2 += f" {'dwA':>5} {'dwP':>4}"
    hdr2 += " | per-round avg"
    print(hdr2)
    print("-" * (110 if has_extra else 90))
    for rank, (combo, avg, scores) in enumerate(deep_results):
        round_avgs = {}
        for sd, sc in zip(filtered_seeds, scores):
            rn = sd["round"]
            if rn not in round_avgs:
                round_avgs[rn] = []
            round_avgs[rn].append(sc)
        per_round = " ".join(f"R{rn}={np.mean(scs):.0f}" for rn, scs in sorted(round_avgs.items()))
        if has_food:
            sp, th, db, df, pgr, fbr, fc, fd = combo
            print(f"{rank+1:4d} | {avg:5.1f} | {sp:.2f} {th:.1f} {db:.3f} {df:.2f} {pgr:.2f}  {fbr:.2f} {fc:.3f} {fd:.3f} | {per_round}")
        elif has_raiding:
            sp, th, db, df, pgr, fbr, rbp, rrng = combo
            print(f"{rank+1:4d} | {avg:5.1f} | {sp:.2f} {th:.1f} {db:.3f} {df:.2f} {pgr:.2f}  {fbr:.2f} {rbp:.3f}  {rrng:>2} | {per_round}")
        elif has_waves:
            sp, th, db, df, pgr, fbr, dwa, dwp = combo
            print(f"{rank+1:4d} | {avg:5.1f} | {sp:.2f} {th:.1f} {db:.3f} {df:.2f} {pgr:.2f}  {fbr:.2f} {dwa:.2f}  {dwp:.1f} | {per_round}")
        else:
            sp, th, db, df, pgr, fbr = combo
            print(f"{rank+1:4d} | {avg:5.1f} | {sp:.2f} {th:.1f} {db:.3f} {df:.2f} {pgr:.2f}  {fbr:.2f} | {per_round}")

    # Output the best params as code
    best_combo = deep_results[0][0]
    print(f"\n=== Best {args.round_type.upper()} params ===")
    print(f"SimParams(")
    if has_food:
        sp, th, db, df, pgr, fbr, fc, fd = best_combo
        print(f"    spawn_prob={sp},")
        print(f"    spawn_pop_threshold={th},")
        print(f"    death_base_rate={db},")
        print(f"    death_food_factor={df},")
        print(f"    pop_growth_rate={pgr},")
        print(f"    food_base_regen={fbr},")
        print(f"    food_competition={fc},")
        print(f"    food_pop_drain={fd},")
        print(f"    spawn_max_per_step=40,")
        print(f"    port_prob=0.10,")
        print(f"    wealth_decay=0.0,")
        print(f"    wealth_coastal_rate=0.01,")
    elif has_waves:
        sp, th, db, df, pgr, fbr, dwa, dwp = best_combo
        print(f"    spawn_prob={sp},")
        print(f"    spawn_pop_threshold={th},")
        print(f"    death_base_rate={db},")
        print(f"    death_food_factor={df},")
        print(f"    pop_growth_rate={pgr},")
        print(f"    food_base_regen={fbr},")
        print(f"    spawn_max_per_step=40,")
        print(f"    port_prob=0.10,")
        print(f"    wealth_decay=0.0,")
        print(f"    wealth_coastal_rate=0.01,")
        print(f"    death_wave_amplitude={dwa},")
        print(f"    death_wave_period={dwp},")
    elif has_raiding:
        sp, th, db, df, pgr, fbr, rbp, rrng = best_combo
        print(f"    spawn_prob={sp},")
        print(f"    spawn_pop_threshold={th},")
        print(f"    death_base_rate={db},")
        print(f"    death_food_factor={df},")
        print(f"    pop_growth_rate={pgr},")
        print(f"    food_base_regen={fbr},")
        print(f"    spawn_max_per_step=40,")
        print(f"    port_prob=0.10,")
        print(f"    wealth_decay=0.0,")
        print(f"    wealth_coastal_rate=0.01,")
        print(f"    raid_base_prob={rbp},")
        print(f"    raid_range={rrng},")
    else:
        sp, th, db, df, pgr, fbr = best_combo
        print(f"    spawn_prob={sp},")
        print(f"    spawn_pop_threshold={th},")
        print(f"    death_base_rate={db},")
        print(f"    death_food_factor={df},")
        print(f"    pop_growth_rate={pgr},")
        print(f"    food_base_regen={fbr},")
        print(f"    spawn_max_per_step=40,")
        print(f"    port_prob=0.10,")
        print(f"    wealth_decay=0.0,")
        print(f"    wealth_coastal_rate=0.01,")
    print(f")")


if __name__ == "__main__":
    main()
