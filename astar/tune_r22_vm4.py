#!/usr/bin/env python3
"""R22 all-in likelihood tuning. Hierarchical: coarse→fine→submit."""

import json, os, sys, time, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from simulator import AstarSimulator
from sim_params import SimParams

ROUND_ID = "a8be24e1-bd48-49bb-aa46-c5593da79f6f"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3ZjdkNzAxZC0xN2I4LTQxYTgtODhlYi05NzYzYjU0MWQ1NTQiLCJlbWFpbCI6ImVybWVzam9hbmRyZWFzQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NjI1NTY0fQ.A-0e_Ga8GZkUbb4NoYJM2Ng-HyLn_B0GeW9Y05KO5ic"
CODE_TO_IDX = {0: 0, 11: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 5}

def load_data():
    with open("replay_cache/round22_detail.json") as f:
        detail = json.load(f)
    with open(f"obs_cache/{ROUND_ID}.json") as f:
        obs = json.load(f)
    # Group by seed
    obs_by_seed = {}
    for o in obs:
        obs_by_seed.setdefault(o["seed"], []).append(o)
    return detail, obs, obs_by_seed

def run_mc_seed(state, params, seed_idx, n_runs):
    """Run MC for one seed, return prob array."""
    grid = np.array(state["grid"])
    sett = [{"x": s["x"], "y": s["y"], "population": s.get("population", 0.5),
             "food": s.get("food", 0.5), "wealth": s.get("wealth", 0.0),
             "defense": s.get("defense", 0.2), "has_port": s.get("has_port", False),
             "owner_id": s.get("owner_id", 0), "alive": True}
            for s in state.get("settlements", []) if s.get("alive", True)]
    sim = AstarSimulator(grid, sett, params=params, seed=42+seed_idx)
    return sim.monte_carlo(n_runs=n_runs, n_steps=50, n_workers=0)

def compute_ll(probs_by_seed, obs_by_seed, W, H):
    """Log-likelihood of observations given MC probs."""
    total_ll = 0.0
    total_cells = 0
    for si, obs_list in obs_by_seed.items():
        if si not in probs_by_seed:
            continue
        probs = probs_by_seed[si]
        for obs in obs_list:
            vp = obs["viewport"]
            ogrid = np.array(obs["grid"])
            for dy in range(ogrid.shape[0]):
                for dx in range(ogrid.shape[1]):
                    gy = vp["y"] + dy
                    gx = vp["x"] + dx
                    if 0 <= gy < H and 0 <= gx < W:
                        tidx = CODE_TO_IDX.get(int(ogrid[dy, dx]), 0)
                        p = max(probs[gy, gx, tidx], 1e-6)
                        total_ll += np.log(p)
                        total_cells += 1
    return total_ll / max(total_cells, 1)

def eval_single_seed(param_dict, detail, obs_by_seed, n_runs, seed_idx=0):
    """Fast eval: 1 seed only."""
    sp = SimParams(**param_dict)
    probs = run_mc_seed(detail["initial_states"][seed_idx], sp, seed_idx, n_runs)
    W, H = detail["map_width"], detail["map_height"]
    sub_obs = {seed_idx: obs_by_seed.get(seed_idx, [])}
    return compute_ll({seed_idx: probs}, sub_obs, W, H)

def eval_all_seeds(param_dict, detail, obs_by_seed, n_runs):
    """Full eval: all 5 seeds."""
    sp = SimParams(**param_dict)
    W, H = detail["map_width"], detail["map_height"]
    probs_by_seed = {}
    for si in range(5):
        probs_by_seed[si] = run_mc_seed(detail["initial_states"][si], sp, si, n_runs)
    return compute_ll(probs_by_seed, obs_by_seed, W, H)

def main():
    print(f"=== R22 ALL-IN LIKELIHOOD TUNING ===")
    print(f"CPUs: {os.cpu_count()}")
    detail, obs, obs_by_seed = load_data()
    W, H = detail["map_width"], detail["map_height"]
    print(f"R22: {len(obs)} obs, 5 seeds, settle_rate~0.047")
    
    # ---- Phase 1: Coarse search, seed 0 only, 20 MC ----
    print("\n--- Phase 1: Coarse (seed 0, 20 MC) ---")
    grid = {
        "spawn_prob": [0.08, 0.10, 0.12, 0.15],
        "spawn_pop_threshold": [0.3, 0.5, 0.7],
        "death_base_rate": [0.02, 0.04, 0.06, 0.08, 0.10],
        "death_food_factor": [0.04, 0.06, 0.08, 0.10],
        "food_base_regen": [0.08, 0.10, 0.12, 0.15],
        "food_competition": [0.03, 0.05, 0.07],
        "food_pop_drain": [0.04, 0.06, 0.08],
        "pop_growth_rate": [0.08, 0.10, 0.12],
        "food_forest_bonus": [0.03],
        "port_prob": [0.10],
        "port_survival_bonus": [0.0],
        "port_wealth_threshold": [0.0],
        "spawn_max_per_step": [40],
    }
    
    keys = list(grid.keys())
    all_combos = list(itertools.product(*[grid[k] for k in keys]))
    total_possible = len(all_combos)
    
    # Sample 800 random combos for speed
    MAX_COARSE = 800
    if total_possible > MAX_COARSE:
        np.random.seed(271)
        idx = np.random.choice(total_possible, MAX_COARSE, replace=False)
        combos = [all_combos[i] for i in idx]
    else:
        combos = all_combos
    
    param_dicts = [{keys[i]: v for i, v in enumerate(c)} for c in combos]
    # Also add interpolated point and nearby calibration
    interp_point = {"spawn_prob": 0.12, "death_base_rate": 0.04, "death_food_factor": 0.06,
                    "food_base_regen": 0.10, "food_competition": 0.06, "food_pop_drain": 0.07,
                    "pop_growth_rate": 0.10, "spawn_pop_threshold": 0.5, "food_forest_bonus": 0.03,
                    "port_prob": 0.10, "port_survival_bonus": 0.0, "port_wealth_threshold": 0.0,
                    "spawn_max_per_step": 40}
    param_dicts.insert(0, interp_point)
    
    total = len(param_dicts)
    print(f"  {total} combos (from {total_possible} possible)")
    
    results = []
    t0 = time.time()
    for i, pd in enumerate(param_dicts):
        ll = eval_single_seed(pd, detail, obs_by_seed, 20, seed_idx=0)
        results.append((ll, pd))
        if (i+1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (total - i - 1)
            best = max(results, key=lambda x: x[0])
            print(f"  [{i+1:>4}/{total}] {elapsed:.0f}s ETA {eta:.0f}s best_ll={best[0]:.4f}")
            sys.stdout.flush()
    
    results.sort(key=lambda x: -x[0])
    print(f"\nPhase 1 done in {time.time()-t0:.0f}s. Top 5:")
    for ll, p in results[:5]:
        print(f"  LL={ll:.4f} sp={p['spawn_prob']} db={p['death_base_rate']} dff={p['death_food_factor']} fbr={p['food_base_regen']} fc={p['food_competition']}")
    
    # ---- Phase 2: Top 30, all 5 seeds, 50 MC ----
    print("\n--- Phase 2: Top 30, all seeds, 50 MC ---")
    top30 = [p for _, p in results[:30]]
    
    results2 = []
    t0 = time.time()
    for i, pd in enumerate(top30):
        ll = eval_all_seeds(pd, detail, obs_by_seed, 50)
        results2.append((ll, pd))
        if (i+1) % 5 == 0:
            best = max(results2, key=lambda x: x[0])
            print(f"  [{i+1}/{len(top30)}] {time.time()-t0:.0f}s best_ll={best[0]:.4f}")
            sys.stdout.flush()
    
    results2.sort(key=lambda x: -x[0])
    print(f"\nPhase 2 done. Top 5:")
    for ll, p in results2[:5]:
        print(f"  LL={ll:.4f} sp={p['spawn_prob']} db={p['death_base_rate']} dff={p['death_food_factor']} fbr={p['food_base_regen']}")
    
    # ---- Phase 3: Fine search around top 3 ----
    print("\n--- Phase 3: Fine search, 100 MC ---")
    fine_params = []
    seen = set()
    for _, base_p in results2[:3]:
        for key in ["spawn_prob", "death_base_rate", "death_food_factor", "food_base_regen",
                     "food_competition", "food_pop_drain", "pop_growth_rate", "spawn_pop_threshold"]:
            for delta in [-0.02, -0.01, -0.005, 0.005, 0.01, 0.02]:
                variant = dict(base_p)
                variant[key] = round(max(0.001, base_p[key] + delta), 4)
                kt = tuple(sorted(variant.items()))
                if kt not in seen:
                    seen.add(kt)
                    fine_params.append(variant)
    
    print(f"  {len(fine_params)} fine combos")
    best_ll = results2[0][0]
    best_params = results2[0][1]
    
    t0 = time.time()
    for i, pd in enumerate(fine_params):
        ll = eval_all_seeds(pd, detail, obs_by_seed, 100)
        if ll > best_ll:
            best_ll = ll
            best_params = pd
            print(f"  NEW BEST at [{i+1}]: LL={ll:.4f}")
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(fine_params)}] {time.time()-t0:.0f}s best_ll={best_ll:.4f}")
            sys.stdout.flush()
    
    print(f"\n=== FINAL BEST ===")
    print(f"LL={best_ll:.4f}")
    print(json.dumps(best_params, indent=2))
    
    # Save
    with open("r22_tuned_params.json", "w") as f:
        json.dump({"best_ll": best_ll, "best_params": best_params}, f, indent=2)
    
    # ---- Phase 4: Big MC + re-submit ----
    print("\n--- Phase 4: 2000 MC + re-submit ---")
    import requests
    headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
    
    sp = SimParams(**best_params)
    states = detail["initial_states"]
    
    from solver import AstarSolver
    solver = AstarSolver(TOKEN, use_mc=True)
    solver._observed_settle_rate = 0.047
    transitions = solver.learn_transitions(obs, states)
    empirical = solver.build_empirical_distributions(obs, states, W, H)
    
    for si in range(5):
        mc = run_mc_seed(states[si], sp, si, 2000)
        sr = mc[:,:,1].mean()
        print(f"  Seed {si}: 2000 MC, settle={sr:.4f}")
        
        analysis = solver.analyse_seed(states[si], W, H)
        pred = solver.build_prediction(si, analysis, obs, transitions, W, H,
                                       mc_pred=mc, empirical=empirical)
        
        try:
            r = requests.post(
                f"https://api.ainm.no/astar-island/rounds/{ROUND_ID}/seeds/{si}/predict",
                headers=headers, json={"prediction": pred.tolist()})
            print(f"  Seed {si}: submitted ({r.status_code})")
        except Exception as e:
            print(f"  Seed {si}: FAILED {e}")
    
    print(f"\n=== R22 RE-SUBMITTED with likelihood-tuned params! ===")
    print(f"Best params: sp={best_params['spawn_prob']} db={best_params['death_base_rate']} fbr={best_params['food_base_regen']}")

if __name__ == "__main__":
    main()
