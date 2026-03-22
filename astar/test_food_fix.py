"""Test food model fix with MC backtest on R20."""
import json, numpy as np, sys, time
sys.path.insert(0, ".")
from simulator import AstarSimulator, SimParams
from backtest import score_prediction
from solver import params_from_rate

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

# Load ground truth
with open("replay_cache/round20_seed0_analysis.json") as f:
    analysis = json.load(f)
gt = np.array(analysis["ground_truth"])

print("R20: %d initial settlements, grid %s" % (len(sett), grid.shape))
print()

# 1. Baseline: current params_from_rate(0.039)
print("=== Baseline (current params_from_rate(0.039)) ===")
base_params = params_from_rate(0.039)
t0 = time.time()
sim = AstarSimulator(grid, sett, params=base_params, seed=42)
probs_base = sim.monte_carlo(n_runs=500, n_steps=50, n_workers=0)
score_base = score_prediction(probs_base, gt)
t1 = time.time()
print("Score: %.1f (%.1fs)" % (score_base, t1-t0))

# Show food/count for baseline
sim_b = AstarSimulator(grid, sett, params=base_params, seed=42)
sim_b.run(50)
print("  Single-run food=%.3f, count=%d" % (sim_b.s_food.mean(), len(sim_b.s_x)))
print()

# 2. Calibrated params: fbr=0.25, fpd=0.04, ff=0.00, db=0.06, dff=0.20
print("=== Calibrated (fbr=0.25 fpd=0.04 ff=0.00 db=0.06 dff=0.20) ===")
base_dict = {k: getattr(base_params, k) for k in vars(base_params)}
base_dict["food_base_regen"] = 0.25
base_dict["food_pop_drain"] = 0.04
base_dict["food_floor"] = 0.00
base_dict["death_base_rate"] = 0.06
base_dict["death_food_factor"] = 0.20
cal_params = SimParams(**base_dict)
t0 = time.time()
sim = AstarSimulator(grid, sett, params=cal_params, seed=42)
probs_cal = sim.monte_carlo(n_runs=500, n_steps=50, n_workers=0)
score_cal = score_prediction(probs_cal, gt)
t1 = time.time()
print("Score: %.1f (%.1fs)" % (score_cal, t1-t0))

sim_c = AstarSimulator(grid, sett, params=cal_params, seed=42)
sim_c.run(50)
print("  Single-run food=%.3f, count=%d" % (sim_c.s_food.mean(), len(sim_c.s_x)))
print()

# 3. Also test top 3 from calibration
configs = [
    ("Top2: fbr=0.25 fpd=0.08 ff=0.10 db=0.06 dff=0.20", 0.25, 0.08, 0.10, 0.06, 0.20),
    ("Top3: fbr=0.30 fpd=0.08 ff=0.20 db=0.08 dff=0.15", 0.30, 0.08, 0.20, 0.08, 0.15),
    ("Top4: fbr=0.30 fpd=0.02 ff=0.20 db=0.08 dff=0.05", 0.30, 0.02, 0.20, 0.08, 0.05),
]

for name, fbr, fpd, ff, db, dff in configs:
    print("=== %s ===" % name)
    d = dict(base_dict)
    d["food_base_regen"] = fbr
    d["food_pop_drain"] = fpd
    d["food_floor"] = ff
    d["death_base_rate"] = db
    d["death_food_factor"] = dff
    p = SimParams(**d)
    t0 = time.time()
    sim = AstarSimulator(grid, sett, params=p, seed=42)
    probs = sim.monte_carlo(n_runs=500, n_steps=50, n_workers=0)
    sc = score_prediction(probs, gt)
    t1 = time.time()
    print("Score: %.1f (%.1fs)" % (sc, t1-t0))
    sim2 = AstarSimulator(grid, sett, params=p, seed=42)
    sim2.run(50)
    print("  Single-run food=%.3f, count=%d" % (sim2.s_food.mean(), len(sim2.s_x)))
    print()

print("\n=== Summary ===")
print("Baseline R20 score: %.1f" % score_base)
print("Calibrated R20 score: %.1f" % score_cal)
print("Delta: %+.1f" % (score_cal - score_base))
