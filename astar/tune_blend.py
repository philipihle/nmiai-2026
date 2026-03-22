"""Grid-search for optimal blend weights (MC/prior/ML) per round type.

Pre-computes prior and MC predictions for all rounds, then searches over
weight combinations to find the best blend. Runs on VM with cached replay data.

Usage:
  python tune_blend.py                    # full search
  python tune_blend.py --mc-runs 500      # more MC runs
  python tune_blend.py --type normal      # only normal rounds
"""

import argparse
import json
import os
import time
from itertools import product

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3ZjdkNzAxZC0xN2I4LTQxYTgtODhlYi05NzYzYjU0MWQ1NTQiLCJlbWFpbCI6ImVybWVzam9hbmRyZWFzQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NjI1NTY0fQ.A-0e_Ga8GZkUbb4NoYJM2Ng-HyLn_B0GeW9Y05KO5ic"

REPLAY_DIR = os.path.join(os.path.dirname(__file__) or ".", "replay_cache")

ALL_ROUNDS = [
    (1, "71451d74-be9f-471f-aacd-a41f3b68a9cd"),
    (2, "76909e29-f664-4b2f-b16b-61b7507277e9"),
    (3, "f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb"),
    (4, "8e839974-b13b-407b-a5e7-fc749d877195"),
    (5, "fd3c92ff-3178-4dc9-8d9b-acf389b3982b"),
    (6, "ae78003a-4efe-425a-881a-d16a39bca0ad"),
    (7, "36e581f1-73f8-453f-ab98-cbe3052b701b"),
    (8, "c5cdf100-a876-4fb7-b5d8-757162c97989"),
    (9, "2a341ace-0f57-4309-9b89-e59fe0f09179"),
    (10, "75e625c3-60cb-4392-af3e-c86a98bde8c2"),
    (11, "324fde07-1670-4202-b199-7aa92ecb40ee"),
    (12, "795bfb1f-54bd-4f39-a526-9868b36f7ebd"),
    (13, "7b4bda99-6165-4221-97cc-27880f5e6d95"),
    (14, "d0a2c894-2162-4d49-86cf-435b9013f3b8"),
]

DIEOFF_ROUNDS = {3, 8, 10}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_prediction(prediction, ground_truth):
    eps = 1e-10
    entropy = -np.sum(ground_truth * np.log(ground_truth + eps), axis=-1)
    kl = np.sum(ground_truth * np.log((ground_truth + eps) / (prediction + eps)), axis=-1)
    total_entropy = entropy.sum()
    if total_entropy < eps:
        return 100.0
    weighted_kl = (entropy * kl).sum() / total_entropy
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_round_from_cache(rnum):
    """Load initial states and GT from replay_cache."""
    detail_file = os.path.join(REPLAY_DIR, f"round{rnum}_detail.json")
    detail = json.load(open(detail_file))

    seeds = []
    for seed_idx in range(5):
        analysis_file = os.path.join(REPLAY_DIR, f"round{rnum}_seed{seed_idx}_analysis.json")
        analysis = json.load(open(analysis_file))
        seeds.append({
            "initial_grid": np.array(analysis["initial_grid"]),
            "ground_truth": np.array(analysis["ground_truth"]),
        })
    return detail, seeds


# ---------------------------------------------------------------------------
# Pre-compute predictions
# ---------------------------------------------------------------------------
def precompute_predictions(rounds, mc_runs=500):
    """Pre-compute prior, MC, and ML predictions for all seeds in all rounds."""
    from solver import AstarSolver
    from sim_params import NORMAL_PARAMS, DIEOFF_PARAMS

    solver = AstarSolver(token=TOKEN, use_mc=True)
    results = {}

    for rnum, rid in rounds:
        round_type = "dieoff" if rnum in DIEOFF_ROUNDS else "normal"
        solver.round_type = round_type
        sim_params = DIEOFF_PARAMS if round_type == "dieoff" else NORMAL_PARAMS

        print(f"\nPrecomputing R{rnum} ({round_type})...")
        detail, seeds_data = load_round_from_cache(rnum)
        W = detail["map_width"]
        H = detail["map_height"]

        round_results = []
        for seed_idx, seed_data in enumerate(seeds_data):
            init_grid = seed_data["initial_grid"]
            gt = seed_data["ground_truth"]

            # Build state dict
            state = {"grid": init_grid.tolist(), "settlements": []}
            for y in range(H):
                for x in range(W):
                    v = int(init_grid[y, x])
                    if v in (1, 2):
                        state["settlements"].append({
                            "x": x, "y": y,
                            "has_port": v == 2,
                            "alive": True,
                        })

            analysis = solver.analyse_seed(state, W, H)

            # Prior-only prediction
            prior_pred = solver.build_prediction(seed_idx, analysis, [], {}, W, H)

            # MC prediction
            t0 = time.time()
            mc_pred = solver._run_mc_prediction(analysis, sim_params, n_runs=mc_runs)
            elapsed = time.time() - t0
            print(f"  Seed {seed_idx}: MC {mc_runs} runs in {elapsed:.1f}s")

            # ML prediction
            ml_pred = None
            if solver.ml_models is not None:
                try:
                    from train_model import compute_features
                    features = compute_features(init_grid, state["settlements"], W, H)
                    ml_pred = np.column_stack([m.predict(features) for m in solver.ml_models])
                    ml_pred = np.maximum(ml_pred, 0.001)
                    ml_pred = ml_pred / ml_pred.sum(axis=-1, keepdims=True)
                    ml_pred = ml_pred.reshape(H, W, 6)
                except Exception as e:
                    print(f"  ML failed: {e}")

            # Precompute masks
            dist = analysis["dist"]
            grid = analysis["grid"]
            dynamic_mask = (dist <= 6) & (grid != 10) & (grid != 5)

            round_results.append({
                "gt": gt,
                "prior": prior_pred,
                "mc": mc_pred,
                "ml": ml_pred,
                "dynamic_mask": dynamic_mask,
                "grid": grid,
                "dist": dist,
            })

        results[rnum] = round_results

    return results


# ---------------------------------------------------------------------------
# Blend + score
# ---------------------------------------------------------------------------
def blend_and_score(data, mc_w, prior_w, ml_w, wf_factor):
    """Blend predictions with given weights and score against GT."""
    H, W = data["gt"].shape[:2]
    prior = data["prior"]
    mc = data["mc"]
    ml = data["ml"]
    dynamic_mask = data["dynamic_mask"]
    grid = data["grid"]

    # Start with prior
    pred = prior.copy()

    if mc is not None and mc_w > 0:
        effective_ml_w = ml_w if ml is not None else 0.0
        total_w = mc_w + prior_w + effective_ml_w
        norm_mc = mc_w / total_w
        norm_prior = prior_w / total_w
        norm_ml = effective_ml_w / total_w

        blend = norm_mc * mc + norm_prior * prior
        if ml is not None and norm_ml > 0:
            blend += norm_ml * ml

        mask_3d = dynamic_mask[:, :, np.newaxis]
        pred = np.where(mask_3d, blend, pred)
    elif ml is not None and ml_w > 0:
        pred = (1 - ml_w) * pred + ml_w * ml

    # Wavefront smoothing
    if wf_factor > 0:
        settle_prob = pred[:, :, 1].copy()
        boost = np.zeros_like(settle_prob)
        if H > 1:
            boost[1:] += settle_prob[:-1]
            boost[:-1] += settle_prob[1:]
        if W > 1:
            boost[:, 1:] += settle_prob[:, :-1]
            boost[:, :-1] += settle_prob[:, 1:]
        boost *= wf_factor
        not_static = (grid != 10) & (grid != 5)
        pred[:, :, 1] += boost * not_static
        pred[:, :, 0] -= boost * not_static * 0.7
        pred[:, :, 4] -= boost * not_static * 0.3

    # Floor and normalize
    pred = np.maximum(pred, 0.001)
    pred /= pred.sum(axis=-1, keepdims=True)
    return score_prediction(pred, data["gt"])


def grid_search(precomputed, round_type, mc_steps=11, ml_steps=7, wf_steps=5):
    """Grid-search over blend weights for a given round type."""
    rnums = [rn for rn in precomputed
             if (round_type == "dieoff") == (rn in DIEOFF_ROUNDS)]

    if not rnums:
        print(f"No {round_type} rounds to search!")
        return

    # Search grid
    mc_values = np.linspace(0, 0.50, mc_steps)
    ml_values = np.linspace(0, 0.30, ml_steps)
    wf_values = np.linspace(0, 0.04, wf_steps)

    if round_type == "dieoff":
        mc_values = np.linspace(0, 0.60, mc_steps)
        ml_values = [0.0]  # ML disabled for die-off

    combos = list(product(mc_values, ml_values, wf_values))
    print(f"\n{'='*60}")
    print(f"Grid search: {round_type.upper()} ({len(rnums)} rounds, {len(combos)} combos)")
    print(f"{'='*60}")

    best_score = -1
    best_params = None
    results = []

    for i, (mc_w, ml_w, wf) in enumerate(combos):
        prior_w = 1.0 - mc_w - ml_w
        if prior_w < 0:
            continue

        round_scores = []
        for rnum in rnums:
            seed_scores = []
            for data in precomputed[rnum]:
                s = blend_and_score(data, mc_w, prior_w, ml_w, wf)
                seed_scores.append(s)
            round_scores.append(np.mean(seed_scores))

        avg = np.mean(round_scores)
        results.append((avg, mc_w, prior_w, ml_w, wf, round_scores))

        if avg > best_score:
            best_score = avg
            best_params = (mc_w, prior_w, ml_w, wf)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(combos)} tested, best so far: {best_score:.2f}")

    # Sort and show top results
    results.sort(key=lambda x: -x[0])

    print(f"\nTop 15 {round_type} configurations:")
    print(f"{'rank':>4} | {'avg':>6} | {'mc_w':>5} {'pr_w':>5} {'ml_w':>5} {'wf':>5} | per-round")
    print(f"{'-'*80}")

    for rank, (avg, mc_w, pr_w, ml_w, wf, rscores) in enumerate(results[:15], 1):
        rstr = " ".join(f"R{rn}={s:.0f}" for rn, s in zip(rnums, rscores))
        print(f"{rank:4d} | {avg:6.2f} | {mc_w:.3f} {pr_w:.3f} {ml_w:.3f} {wf:.3f} | {rstr}")

    print(f"\n=== Best {round_type} params ===")
    mc_w, pr_w, ml_w, wf = best_params
    print(f"  mc_weight = {mc_w:.3f}")
    print(f"  prior_weight = {pr_w:.3f}")
    print(f"  ml_weight = {ml_w:.3f}")
    print(f"  wavefront_factor = {wf:.4f}")
    print(f"  score = {best_score:.2f}")

    return best_params, best_score


def main():
    parser = argparse.ArgumentParser(description="Tune blend weights")
    parser.add_argument("--mc-runs", type=int, default=500, help="MC simulation runs")
    parser.add_argument("--type", choices=["normal", "dieoff", "both"], default="both")
    args = parser.parse_args()

    t0 = time.time()

    # Pre-compute all predictions
    print("Pre-computing predictions...")
    precomputed = precompute_predictions(ALL_ROUNDS, mc_runs=args.mc_runs)
    print(f"\nPre-computation done in {time.time() - t0:.0f}s")

    # Grid search
    if args.type in ("normal", "both"):
        grid_search(precomputed, "normal")

    if args.type in ("dieoff", "both"):
        grid_search(precomputed, "dieoff")

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
