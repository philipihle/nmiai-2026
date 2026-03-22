"""Tuned SimParams for normal and die-off rounds.

Found by tune_sim.py on 192-CPU VM (2026-03-21):
  Normal (d=2 spawn): 5400 combos, 30 seeds, 200 MC deep eval → 77.6 MC-only
  Dieoff: 2700 combos, 10 seeds (R3+R8), 200 MC deep eval → 85.5 MC-only

Update by re-running: python tune_sim.py --round-type normal|dieoff
"""
from simulator import SimParams

NORMAL_PARAMS = SimParams(
    spawn_prob=0.08,
    spawn_pop_threshold=0.3,
    death_base_rate=0.005,
    death_food_factor=0.04,
    pop_growth_rate=0.12,
    food_base_regen=0.20,
    food_competition=0.03,
    food_pop_drain=0.06,
    food_forest_bonus=0.06,
    port_wealth_threshold=0.0,
    spawn_max_per_step=40,
    port_prob=0.10,
    wealth_decay=0.0,
    wealth_coastal_rate=0.01,
)

DIEOFF_PARAMS = SimParams(
    spawn_prob=0.15,
    spawn_pop_threshold=0.5,
    death_base_rate=0.10,
    death_food_factor=0.10,
    pop_growth_rate=0.08,
    food_base_regen=0.10,
    spawn_max_per_step=40,
    port_prob=0.10,
    wealth_decay=0.0,
    wealth_coastal_rate=0.01,
)
