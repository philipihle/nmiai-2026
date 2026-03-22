# calibrate_food.py
import json, numpy as np, sys, itertools
sys.path.insert(0, ".")
from simulator import AstarSimulator, SimParams
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

# Replay targets for R20
TARGET_FOOD = 0.70
TARGET_DEATHS_PER_STEP = 3.3
TARGET_FINAL_SETTLEMENTS = 55

# Get base params for R20
base = params_from_rate(0.039)
base_dict = {k: getattr(base, k) for k in [
    "spawn_prob","spawn_pop_threshold","death_base_rate","death_food_factor",
    "food_base_regen","food_competition","food_pop_drain","pop_growth_rate",
    "food_forest_bonus","port_prob","port_survival_bonus","port_wealth_threshold",
    "spawn_max_per_step"]}

# Test grid
food_regens = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
food_drains = [0.02, 0.04, 0.06, 0.08]
food_floors = [0.0, 0.05, 0.10, 0.20]
death_bases = [0.02, 0.04, 0.06, 0.08, 0.10]
death_food_factors = [0.05, 0.10, 0.15, 0.20]

print("Calibration: finding params to match replay dynamics")
print("Targets: food=%.2f, deaths/step=%.1f, final_settlements=%d" % (TARGET_FOOD, TARGET_DEATHS_PER_STEP, TARGET_FINAL_SETTLEMENTS))
print("Initial settlements: %d" % len(sett))
print("Base params: fbr=%.2f fpd=%.2f db=%.2f dff=%.2f" % (
    base_dict["food_base_regen"], base_dict["food_pop_drain"],
    base_dict["death_base_rate"], base_dict["death_food_factor"]))
print()

best_score = 999
best_params = None
results = []

total = len(food_regens) * len(food_drains) * len(food_floors) * len(death_bases) * len(death_food_factors)
done = 0

for fbr, fpd, ff, db, dff in itertools.product(food_regens, food_drains, food_floors, death_bases, death_food_factors):
    params = dict(base_dict)
    params["food_base_regen"] = fbr
    params["food_pop_drain"] = fpd
    params["food_floor"] = ff
    params["death_base_rate"] = db
    params["death_food_factor"] = dff
    
    sp = SimParams(**params)
    sim = AstarSimulator(grid, sett, params=sp, seed=42)
    
    for step in range(50):
        sim.step()
    
    alive = np.ones(len(sim.s_x), dtype=bool)  # all remaining are alive
    final_food = sim.s_food.mean() if len(sim.s_food) > 0 else 0
    final_count = len(sim.s_x)
    
    # Score: weighted distance from targets
    food_err = abs(final_food - TARGET_FOOD) / TARGET_FOOD
    count_err = abs(final_count - TARGET_FINAL_SETTLEMENTS) / TARGET_FINAL_SETTLEMENTS
    score = food_err + count_err
    
    results.append((score, fbr, fpd, ff, db, dff, final_food, final_count))
    
    if score < best_score:
        best_score = score
        best_params = (fbr, fpd, ff, db, dff)
    
    done += 1
    if done % 200 == 0:
        print("Progress: %d/%d (%.0f%%)" % (done, total, 100*done/total))

# Sort and print top 20
results.sort()
print()
print("Top 20 configurations:")
print("%-6s %-6s %-6s %-6s %-6s %-8s %-8s %-8s" % ("fbr", "fpd", "ff", "db", "dff", "food50", "final", "score"))
for score, fbr, fpd, ff, db, dff, food, count in results[:20]:
    print("%-6.2f %-6.2f %-6.2f %-6.2f %-6.2f %-8.3f %-8d %-8.3f" % (fbr, fpd, ff, db, dff, food, count, score))

print()
print("Best: fbr=%.2f fpd=%.2f ff=%.2f db=%.2f dff=%.2f" % best_params)
print("Best score: %.3f" % best_score)

# Also show configs near target food AND target count
print()
print("Configs with food in [0.60, 0.80] AND final in [45, 65]:")
filtered = [(s, fbr, fpd, ff, db, dff, food, cnt)
            for s, fbr, fpd, ff, db, dff, food, cnt in results
            if 0.60 <= food <= 0.80 and 45 <= cnt <= 65]
print("Found %d matching configs" % len(filtered))
for score, fbr, fpd, ff, db, dff, food, count in filtered[:30]:
    print("%-6.2f %-6.2f %-6.2f %-6.2f %-6.2f %-8.3f %-8d %-8.3f" % (fbr, fpd, ff, db, dff, food, count, score))
