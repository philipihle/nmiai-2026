"""Test updating just the R20 calibration point in solver.py."""
import json, numpy as np, sys, time
sys.path.insert(0, ".")
from simulator import AstarSimulator, SimParams
from backtest import score_prediction

# Load R20 data
with open("replay_cache/round20_detail.json") as f:
    detail = json.load(f)
state = detail["initial_states"][0]
grid = np.array(state["grid"])
sett = [{"x":s["x"],"y":s["y"],"population":s.get("population",0.5),
         "food":s.get("food",0.5),"wealth":s.get("wealth",0.0),
         "defense":s.get("defense",0.2),"has_port":s.get("has_port",False),
         "owner_id":s.get("owner_id",0),"alive":True}
        for s in state.get("settlements",[]) if s.get("alive",True)]

with open("replay_cache/round20_seed0_analysis.json") as f:
    gt = np.array(json.load(f)["ground_truth"])

print("Testing R20 calibration point update")
print("Initial: %d settlements" % len(sett))
print()

# Current R20 calibration
current = {
    "death_base_rate": 0.050,
    "death_food_factor": 0.050,
    "food_base_regen": 0.080,
    "food_competition": 0.070,
    "food_forest_bonus": 0.040,
    "food_pop_drain": 0.060,
    "pop_growth_rate": 0.080,
    "port_prob": 0.080,
    "port_survival_bonus": 0.000,
    "spawn_pop_threshold": 0.700,
    "spawn_prob": 0.150,
}

# Updated R20 calibration (food-dynamics-corrected)
updated = dict(current)
updated["food_base_regen"] = 0.25
updated["food_pop_drain"] = 0.04
updated["death_base_rate"] = 0.06
updated["death_food_factor"] = 0.20

# Test current
sp_cur = SimParams(**current)
t0 = time.time()
sim = AstarSimulator(grid, sett, params=sp_cur, seed=42)
probs_cur = sim.monte_carlo(n_runs=500, n_steps=50, n_workers=0)
score_cur = score_prediction(probs_cur, gt)
print("Current:  score=%.1f (%.1fs)" % (score_cur, time.time()-t0))
sim_c = AstarSimulator(grid, sett, params=sp_cur, seed=42)
sim_c.run(50)
print("  food=%.3f count=%d" % (sim_c.s_food.mean(), len(sim_c.s_x)))

# Test updated
sp_upd = SimParams(**updated)
t0 = time.time()
sim = AstarSimulator(grid, sett, params=sp_upd, seed=42)
probs_upd = sim.monte_carlo(n_runs=500, n_steps=50, n_workers=0)
score_upd = score_prediction(probs_upd, gt)
print("Updated:  score=%.1f (%.1fs)" % (score_upd, time.time()-t0))
sim_u = AstarSimulator(grid, sett, params=sp_upd, seed=42)
sim_u.run(50)
print("  food=%.3f count=%d" % (sim_u.s_food.mean(), len(sim_u.s_x)))

# Test with food_competition and food_forest_bonus tuned too
# The calibrated search didn't tune these; let's also try adjusting them
variants = [
    ("Updated + low competition", {**updated, "food_competition": 0.03}),
    ("Updated + high forest bonus", {**updated, "food_forest_bonus": 0.08}),
    ("Updated + both", {**updated, "food_competition": 0.03, "food_forest_bonus": 0.08}),
    ("Updated + pop_growth=0.10", {**updated, "pop_growth_rate": 0.10}),
    ("Updated + spawn_thresh=0.50", {**updated, "spawn_pop_threshold": 0.50}),
]

print()
for name, params in variants:
    sp = SimParams(**params)
    sim = AstarSimulator(grid, sett, params=sp, seed=42)
    probs = sim.monte_carlo(n_runs=500, n_steps=50, n_workers=0)
    sc = score_prediction(probs, gt)
    sim2 = AstarSimulator(grid, sett, params=sp, seed=42)
    sim2.run(50)
    print("%s: score=%.1f food=%.3f count=%d" % (name, sc, sim2.s_food.mean(), len(sim2.s_x)))

print()
print("Delta (updated - current): %+.1f" % (score_upd - score_cur))
