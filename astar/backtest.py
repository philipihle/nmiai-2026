"""
Backtest the solver's predictions against ground truth from completed rounds.

Modes:
  --mode prior     Prior-only predictions (no MC, no ML) — current baseline
  --mode mc        MC-only predictions (simulator as primary model)
  --mode ensemble  Full ensemble: MC + priors + ML (new default)

Usage:
  python backtest.py                          # all modes, default rounds
  python backtest.py --mode mc                # MC-only
  python backtest.py --mode ensemble          # full ensemble
  python backtest.py --rounds 1 3 7           # specific rounds
"""

import argparse
import numpy as np
import requests
import sys

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3ZjdkNzAxZC0xN2I4LTQxYTgtODhlYi05NzYzYjU0MWQ1NTQiLCJlbWFpbCI6ImVybWVzam9hbmRyZWFzQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NjI1NTY0fQ.A-0e_Ga8GZkUbb4NoYJM2Ng-HyLn_B0GeW9Y05KO5ic"
BASE = "https://api.ainm.no"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# Round classification
DIEOFF_ROUNDS = {3, 8, 10, 19, 20}

# Updated with R11-R14 from auto-solve


def fetch_round_data(round_id):
    """Fetch initial states and ground truth for a completed round."""
    detail = requests.get(f"{BASE}/astar-island/rounds/{round_id}", headers=HEADERS).json()
    seeds = []
    for seed_idx in range(detail["seeds_count"]):
        analysis = requests.get(
            f"{BASE}/astar-island/analysis/{round_id}/{seed_idx}", headers=HEADERS
        ).json()
        seeds.append({
            "initial_grid": np.array(analysis["initial_grid"]),
            "ground_truth": np.array(analysis["ground_truth"]),
        })
    return detail, seeds


def score_prediction(prediction, ground_truth):
    """
    Compute the competition score: entropy-weighted KL divergence.
    score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
    """
    H, W, C = ground_truth.shape
    eps = 1e-10

    # Entropy per cell
    entropy = -np.sum(ground_truth * np.log(ground_truth + eps), axis=-1)

    # KL divergence per cell: sum( p * log(p/q) )
    kl = np.sum(ground_truth * np.log((ground_truth + eps) / (prediction + eps)), axis=-1)

    # Weight by entropy (only score "interesting" cells)
    total_entropy = entropy.sum()
    if total_entropy < eps:
        return 100.0  # all cells are deterministic

    weighted_kl = (entropy * kl).sum() / total_entropy
    score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))
    return score


def score_by_terrain(prediction, ground_truth, initial_grid):
    """Break down score contribution by initial terrain type."""
    eps = 1e-10
    entropy = -np.sum(ground_truth * np.log(ground_truth + eps), axis=-1)
    kl = np.sum(ground_truth * np.log((ground_truth + eps) / (prediction + eps)), axis=-1)

    terrain_names = {10: "Ocean", 11: "Plains", 0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain"}

    print(f"  {'Terrain':>12s} | {'cells':>5s} | {'avg_ent':>7s} | {'avg_KL':>7s} | {'contrib%':>8s}")
    print(f"  {'-'*55}")

    total_weighted = (entropy * kl).sum()

    for tv in [10, 11, 0, 1, 2, 3, 4, 5]:
        mask = initial_grid == tv
        n = mask.sum()
        if n == 0:
            continue
        avg_ent = entropy[mask].mean()
        avg_kl = kl[mask].mean()
        contrib = (entropy[mask] * kl[mask]).sum()
        pct = 100 * contrib / total_weighted if total_weighted > 0 else 0
        name = terrain_names.get(tv, f"?{tv}")
        print(f"  {name:>12s} | {n:5d} | {avg_ent:7.3f} | {avg_kl:7.3f} | {pct:7.1f}%")


def show_worst_cells(prediction, ground_truth, initial_grid, n=10):
    """Show the cells where our prediction is worst."""
    eps = 1e-10
    kl = np.sum(ground_truth * np.log((ground_truth + eps) / (prediction + eps)), axis=-1)
    entropy = -np.sum(ground_truth * np.log(ground_truth + eps), axis=-1)
    weighted = entropy * kl

    terrain_names = {10: "Ocn", 11: "Pln", 0: "Emp", 1: "Set", 2: "Prt", 3: "Rui", 4: "For", 5: "Mtn"}

    flat = weighted.flatten()
    worst_idx = np.argsort(flat)[-n:][::-1]

    print(f"\n  Worst {n} cells (highest weighted KL):")
    print(f"  {'(y,x)':>8s} | {'init':>4s} | {'KL':>6s} | {'ent':>5s} | {'GT distribution':>40s} | {'Our prediction':>40s}")
    print(f"  {'-'*120}")

    H, W = initial_grid.shape
    for idx in worst_idx:
        y, x = divmod(idx, W)
        tv = initial_grid[y, x]
        tname = terrain_names.get(tv, f"?{tv}")
        gt_str = " ".join(f"{v:.2f}" for v in ground_truth[y, x])
        pr_str = " ".join(f"{v:.2f}" for v in prediction[y, x])
        print(f"  ({y:2d},{x:2d}) | {tname:>4s} | {kl[y,x]:6.3f} | {entropy[y,x]:5.3f} | {gt_str:>40s} | {pr_str:>40s}")


def build_mc_prediction(solver, analysis, state, W, H, round_type):
    """Build prediction using MC simulation with calibrated params."""
    from solver import params_from_rate
    settle_rate = getattr(solver, '_observed_settle_rate', 0)
    if settle_rate > 0:
        sim_params = params_from_rate(settle_rate)
    else:
        from sim_params import NORMAL_PARAMS, DIEOFF_PARAMS
        sim_params = DIEOFF_PARAMS if round_type == "dieoff" else NORMAL_PARAMS
    mc_pred = solver._run_mc_prediction(analysis, sim_params, n_runs=500)
    return mc_pred


def main():
    parser = argparse.ArgumentParser(description="Backtest solver predictions")
    parser.add_argument("--mode", choices=["prior", "mc", "ensemble", "all"],
                        default="all", help="Prediction mode to test")
    parser.add_argument("--rounds", nargs="+", type=int, default=None,
                        help="Specific round numbers to test")
    args = parser.parse_args()

    from solver import AstarSolver

    # All completed rounds with known IDs
    ALL_ROUNDS = [
        ("71451d74-be9f-471f-aacd-a41f3b68a9cd", 1),
        ("76909e29-f664-4b2f-b16b-61b7507277e9", 2),
        ("f1dac9a9-5cf1-49a9-8f17-d6cb5d5ba5cb", 3),
        ("8e839974-b13b-407b-a5e7-fc749d877195", 4),
        ("fd3c92ff-3178-4dc9-8d9b-acf389b3982b", 5),
        ("ae78003a-4efe-425a-881a-d16a39bca0ad", 6),
        ("36e581f1-73f8-453f-ab98-cbe3052b701b", 7),
        ("c5cdf100-a876-4fb7-b5d8-757162c97989", 8),
        ("2a341ace-0f57-4309-9b89-e59fe0f09179", 9),
        ("75e625c3-60cb-4392-af3e-c86a98bde8c2", 10),
        ("324fde07-1670-4202-b199-7aa92ecb40ee", 11),
        ("795bfb1f-54bd-4f39-a526-9868b36f7ebd", 12),
        ("7b4bda99-6165-4221-97cc-27880f5e6d95", 13),
        ("d0a2c894-2162-4d49-86cf-435b9013f3b8", 14),
        ("cc5442dd-bc5d-418b-911b-7eb960cb0390", 15),
        ("8f664aed-8839-4c85-bed0-77a2cac7c6f5", 16),
        ("3eb0c25d-28fa-48ca-b8e1-fc249e3918e9", 17),
        ("b0f9d1bf-4b71-4e6e-816c-19c718d29056", 18),
        ("597e60cf-d1a1-4627-ac4d-2a61da68b6df", 19),
        ("fd82f643-15e2-40e7-9866-8d8f5157081c", 20),
    ]

    if args.rounds:
        completed_rounds = [(rid, rn) for rid, rn in ALL_ROUNDS if rn in args.rounds]
    else:
        completed_rounds = ALL_ROUNDS

    if not completed_rounds:
        print("No matching rounds found!")
        return

    modes = ["prior", "mc", "ensemble"] if args.mode == "all" else [args.mode]

    for mode in modes:
        print(f"\n{'#'*60}")
        print(f"  MODE: {mode.upper()}")
        print(f"{'#'*60}")

        solver = AstarSolver(TOKEN, use_mc=(mode != "prior"))
        all_scores = []

        for round_id, round_num in completed_rounds:
            round_type = "dieoff" if round_num in DIEOFF_ROUNDS else "normal"
            solver.round_type = round_type

            print(f"\n{'='*60}")
            print(f"  Round {round_num} ({round_type})")
            print(f"{'='*60}")

            try:
                detail, seeds = fetch_round_data(round_id)
            except Exception as e:
                print(f"  Failed to fetch round {round_num}: {e}")
                continue

            W = detail["map_width"]
            H = detail["map_height"]

            # Compute settle_rate from ground truth for adaptive weighting
            gt_rates = []
            for sd in seeds:
                gt = sd["ground_truth"]
                gt_rates.append(float(gt[:, :, 1].mean() + gt[:, :, 2].mean()))
            solver._observed_settle_rate = np.mean(gt_rates) if gt_rates else 0.0

            scores = []
            for seed_idx, seed_data in enumerate(seeds):
                init_grid = seed_data["initial_grid"]
                gt = seed_data["ground_truth"]

                state = {"grid": init_grid.tolist(), "settlements": []}
                for y in range(H):
                    for x in range(W):
                        v = init_grid[y, x]
                        if v in (1, 2):
                            state["settlements"].append({
                                "x": x, "y": y,
                                "has_port": v == 2,
                                "alive": True,
                            })

                analysis = solver.analyse_seed(state, W, H)

                if mode == "prior":
                    prediction = solver.build_prediction(seed_idx, analysis, [], {}, W, H)
                elif mode == "mc":
                    mc_pred = build_mc_prediction(solver, analysis, state, W, H, round_type)
                    if mc_pred is not None:
                        prediction = mc_pred
                    else:
                        prediction = solver.build_prediction(seed_idx, analysis, [], {}, W, H)
                else:  # ensemble
                    mc_pred = build_mc_prediction(solver, analysis, state, W, H, round_type)
                    prediction = solver.build_prediction(seed_idx, analysis, [], {}, W, H, mc_pred=mc_pred)

                s = score_prediction(prediction, gt)
                scores.append(s)

                print(f"\n--- Seed {seed_idx}: score = {s:.1f} ---")
                score_by_terrain(prediction, gt, init_grid)
                show_worst_cells(prediction, gt, init_grid, n=5)

            avg = np.mean(scores)
            all_scores.extend(scores)
            print(f"\n>>> Round {round_num} average score: {avg:.1f}")
            print(f"    Per-seed: {', '.join(f'{s:.1f}' for s in scores)}")

        if all_scores:
            print(f"\n{'='*60}")
            print(f"  {mode.upper()} OVERALL: {np.mean(all_scores):.1f}")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
