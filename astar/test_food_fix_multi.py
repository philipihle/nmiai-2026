"""Test food fix across multiple rounds to check generalization."""
import json, numpy as np, sys, time, os
sys.path.insert(0, ".")
from simulator import AstarSimulator, SimParams
from backtest import score_prediction, score_by_terrain
from solver import params_from_rate

# Check which rounds have replay data
rounds_available = []
for r in range(1, 23):
    path = f"replay_cache/round{r}_detail.json"
    analysis_path = f"replay_cache/round{r}_seed0_analysis.json"
    if os.path.exists(path) and os.path.exists(analysis_path):
        rounds_available.append(r)

print("Available rounds:", rounds_available)
print()

# Known settle rates per round (from calibration data)
SETTLE_RATES = {
    1: 0.065, 2: 0.063, 3: 0.025, 4: 0.055, 5: 0.060,
    6: 0.058, 7: 0.048, 8: 0.030, 9: 0.044, 10: 0.020,
    11: 0.055, 12: 0.050, 13: 0.045, 14: 0.040, 15: 0.055,
    16: 0.050, 17: 0.048, 18: 0.042, 19: 0.025, 20: 0.039,
    21: 0.050, 22: 0.045,
}

# Test both baseline and calibrated on available dieoff rounds
DIEOFF_ROUNDS = {3, 8, 10, 19, 20}

for r in rounds_available:
    if r not in DIEOFF_ROUNDS and r not in {7, 14, 18}:
        continue  # Focus on dieoff + a few normal rounds
    
    rate = SETTLE_RATES.get(r, 0.045)
    
    with open(f"replay_cache/round{r}_detail.json") as f:
        detail = json.load(f)
    state = detail["initial_states"][0]
    grid = np.array(state["grid"])
    sett = [{"x":s["x"],"y":s["y"],"population":s.get("population",0.5),
             "food":s.get("food",0.5),"wealth":s.get("wealth",0.0),
             "defense":s.get("defense",0.2),"has_port":s.get("has_port",False),
             "owner_id":s.get("owner_id",0),"alive":True}
            for s in state.get("settlements",[]) if s.get("alive",True)]
    
    with open(f"replay_cache/round{r}_seed0_analysis.json") as f:
        analysis = json.load(f)
    gt = np.array(analysis["ground_truth"])
    
    print(f"=== R{r} (rate={rate}, {'DIEOFF' if r in DIEOFF_ROUNDS else 'normal'}, {len(sett)} settlements) ===")
    
    # Baseline
    bp = params_from_rate(rate)
    sim = AstarSimulator(grid, sett, params=bp, seed=42)
    probs = sim.monte_carlo(n_runs=200, n_steps=50, n_workers=0)
    base_score = score_prediction(probs, gt)
    
    # Calibrated
    d = {k: getattr(bp, k) for k in vars(bp)}
    d["food_base_regen"] = 0.25
    d["food_pop_drain"] = 0.04
    d["death_base_rate"] = 0.06
    d["death_food_factor"] = 0.20
    cp = SimParams(**d)
    sim = AstarSimulator(grid, sett, params=cp, seed=42)
    probs = sim.monte_carlo(n_runs=200, n_steps=50, n_workers=0)
    cal_score = score_prediction(probs, gt)
    
    delta = cal_score - base_score
    marker = "+++" if delta > 1 else "---" if delta < -1 else "   "
    print(f"  Baseline: {base_score:.1f}  Calibrated: {cal_score:.1f}  Delta: {delta:+.1f} {marker}")
    
    # Quick dynamics check
    sim_b = AstarSimulator(grid, sett, params=bp, seed=42)
    sim_b.run(50)
    sim_c = AstarSimulator(grid, sett, params=cp, seed=42)
    sim_c.run(50)
    print(f"  Base: food={sim_b.s_food.mean():.3f} count={len(sim_b.s_x)}  Cal: food={sim_c.s_food.mean():.3f} count={len(sim_c.s_x)}")
    print()
