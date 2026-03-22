"""Analyze R12 (score 63.4) to understand what went wrong.

Compares prior-only, MC-only, and ensemble predictions against GT.
Breaks down by terrain, distance, and identifies worst cells.
"""

import json
import os

import numpy as np

REPLAY_DIR = os.path.join(os.path.dirname(__file__) or ".", "replay_cache")
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3ZjdkNzAxZC0xN2I4LTQxYTgtODhlYi05NzYzYjU0MWQ1NTQiLCJlbWFpbCI6ImVybWVzam9hbmRyZWFzQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NjI1NTY0fQ.A-0e_Ga8GZkUbb4NoYJM2Ng-HyLn_B0GeW9Y05KO5ic"

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
TERRAIN_NAMES = {10: "Ocean", 11: "Plains", 0: "Empty", 1: "Settlement",
                 2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain"}


def score_prediction(prediction, ground_truth):
    eps = 1e-10
    entropy = -np.sum(ground_truth * np.log(ground_truth + eps), axis=-1)
    kl = np.sum(ground_truth * np.log((ground_truth + eps) / (prediction + eps)), axis=-1)
    total_entropy = entropy.sum()
    if total_entropy < eps:
        return 100.0
    weighted_kl = (entropy * kl).sum() / total_entropy
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))


def analyze_round(rnum, compare_rounds=None):
    """Full analysis of a round."""
    from solver import AstarSolver
    from sim_params import NORMAL_PARAMS, DIEOFF_PARAMS

    detail = json.load(open(os.path.join(REPLAY_DIR, f"round{rnum}_detail.json")))
    W = detail["map_width"]
    H = detail["map_height"]

    # Check GT settlement rates
    print(f"\n{'='*70}")
    print(f"  ROUND {rnum} ANALYSIS")
    print(f"{'='*70}")

    gts = []
    for seed in range(5):
        gt = np.array(json.load(open(
            os.path.join(REPLAY_DIR, f"round{rnum}_seed{seed}_analysis.json")
        ))["ground_truth"])
        gts.append(gt)
        settle_rate = gt[:, :, 1].mean()
        ruin_rate = gt[:, :, 3].mean()
        print(f"  Seed {seed}: settle={settle_rate:.4f} ruin={ruin_rate:.4f}")

    avg_settle = np.mean([g[:, :, 1].mean() for g in gts])
    is_dieoff = avg_settle < 0.03
    round_type = "dieoff" if is_dieoff else "normal"
    print(f"\n  Avg settlement rate: {avg_settle:.4f} → {round_type.upper()}")

    # Compare with other rounds
    if compare_rounds:
        print(f"\n  Comparison with other rounds (GT settlement rates):")
        for crnum in compare_rounds:
            try:
                cgt = np.array(json.load(open(
                    os.path.join(REPLAY_DIR, f"round{crnum}_seed0_analysis.json")
                ))["ground_truth"])
                print(f"    R{crnum}: settle={cgt[:,:,1].mean():.4f}")
            except Exception:
                pass

    # Build predictions: prior-only, MC-only, ensemble
    solver = AstarSolver(token=TOKEN, use_mc=True)
    solver.round_type = round_type
    sim_params = DIEOFF_PARAMS if is_dieoff else NORMAL_PARAMS

    print(f"\n  Per-seed scores:")
    print(f"  {'seed':>4} | {'prior':>6} | {'mc':>6} | {'ensemble':>8} | {'delta':>6}")
    print(f"  {'-'*45}")

    for seed_idx in range(5):
        init_grid = np.array(json.load(open(
            os.path.join(REPLAY_DIR, f"round{rnum}_seed{seed_idx}_analysis.json")
        ))["initial_grid"])
        gt = gts[seed_idx]

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

        # Prior
        prior_pred = solver.build_prediction(seed_idx, analysis, [], {}, W, H)
        prior_score = score_prediction(prior_pred, gt)

        # MC
        mc_pred = solver._run_mc_prediction(analysis, sim_params, n_runs=500)
        mc_score = score_prediction(mc_pred, gt) if mc_pred is not None else 0

        # Ensemble (current weights)
        ens_pred = solver.build_prediction(seed_idx, analysis, [], {}, W, H, mc_pred=mc_pred)
        ens_score = score_prediction(ens_pred, gt)

        delta = ens_score - prior_score
        print(f"  {seed_idx:4d} | {prior_score:6.1f} | {mc_score:6.1f} | {ens_score:8.1f} | {delta:+6.1f}")

        # Detailed breakdown for worst seed
        if seed_idx == 0 or delta < -5:
            _terrain_breakdown(prior_pred, mc_pred, ens_pred, gt, init_grid, analysis, seed_idx)


def _terrain_breakdown(prior, mc, ens, gt, init_grid, analysis, seed_idx):
    """Show where MC helps/hurts vs prior."""
    eps = 1e-10
    dist = analysis["dist"]

    # Per-terrain KL for prior vs ensemble
    print(f"\n    Seed {seed_idx} terrain breakdown (prior → ensemble):")
    print(f"    {'terrain':>12} | {'cells':>5} | {'prior_KL':>8} | {'ens_KL':>8} | {'delta':>8}")
    print(f"    {'-'*55}")

    kl_prior = np.sum(gt * np.log((gt + eps) / (prior + eps)), axis=-1)
    kl_ens = np.sum(gt * np.log((gt + eps) / (ens + eps)), axis=-1)
    entropy = -np.sum(gt * np.log(gt + eps), axis=-1)

    for tv in [11, 1, 2, 4, 3, 0, 10, 5]:
        mask = init_grid == tv
        n = mask.sum()
        if n == 0:
            continue
        avg_kl_p = (entropy[mask] * kl_prior[mask]).sum() / max(entropy[mask].sum(), eps)
        avg_kl_e = (entropy[mask] * kl_ens[mask]).sum() / max(entropy[mask].sum(), eps)
        name = TERRAIN_NAMES.get(tv, f"?{tv}")
        delta = avg_kl_e - avg_kl_p
        print(f"    {name:>12} | {n:5d} | {avg_kl_p:8.4f} | {avg_kl_e:8.4f} | {delta:+8.4f}")

    # By distance bucket
    print(f"\n    Seed {seed_idx} by distance from settlement:")
    print(f"    {'dist':>8} | {'cells':>5} | {'prior_KL':>8} | {'mc_KL':>8} | {'ens_KL':>8}")
    print(f"    {'-'*55}")

    kl_mc = np.sum(gt * np.log((gt + eps) / (mc + eps)), axis=-1) if mc is not None else np.zeros_like(kl_prior)

    for dmin, dmax, label in [(0, 0, "d=0"), (1, 2, "d=1-2"), (3, 4, "d=3-4"),
                               (5, 7, "d=5-7"), (8, 999, "d=8+")]:
        mask = (dist >= dmin) & (dist <= dmax) & (init_grid != 10) & (init_grid != 5)
        n = mask.sum()
        if n == 0:
            continue
        e = entropy[mask]
        if e.sum() < eps:
            continue
        p_kl = (e * kl_prior[mask]).sum() / e.sum()
        m_kl = (e * kl_mc[mask]).sum() / e.sum()
        e_kl = (e * kl_ens[mask]).sum() / e.sum()
        print(f"    {label:>8} | {n:5d} | {p_kl:8.4f} | {m_kl:8.4f} | {e_kl:8.4f}")

    # Worst 5 cells
    weighted = entropy * kl_ens
    flat = weighted.flatten()
    worst_idx = np.argsort(flat)[-5:][::-1]
    H, W = init_grid.shape

    print(f"\n    Worst 5 cells (ensemble):")
    print(f"    {'(y,x)':>8} | {'init':>4} | {'dist':>4} | {'GT':>35} | {'prior':>35} | {'ens':>35}")
    print(f"    {'-'*135}")

    for idx in worst_idx:
        y, x = divmod(idx, W)
        tv = int(init_grid[y, x])
        tname = TERRAIN_NAMES.get(tv, "?")[:3]
        d = dist[y, x]
        gt_str = " ".join(f"{v:.2f}" for v in gt[y, x])
        pr_str = " ".join(f"{v:.2f}" for v in prior[y, x])
        en_str = " ".join(f"{v:.2f}" for v in ens[y, x])
        print(f"    ({y:2d},{x:2d}) | {tname:>4} | {d:4.0f} | {gt_str:>35} | {pr_str:>35} | {en_str:>35}")


def main():
    # Analyze R12 specifically
    analyze_round(12, compare_rounds=[1, 2, 4, 5, 6, 7, 9, 11, 13, 14])

    # Also check R9 seed 4 (59.7)
    print("\n\n" + "#" * 70)
    print("  R9 SEED 4 (scored 59.7)")
    print("#" * 70)
    analyze_round(9)


if __name__ == "__main__":
    main()
