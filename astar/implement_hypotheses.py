"""Implement confirmed hypotheses into simulator_exp.py.

Reads hypothesis_results.json, patches ONLY simulator_exp.py.
Never touches production simulator.py.

Usage:
  python implement_hypotheses.py
"""
import json
import os
import re
import sys


def load_results():
    with open("hypothesis_results.json") as f:
        return json.load(f)


def read_file(path):
    with open(path) as f:
        return f.read()


def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def add_sim_param(content, param_name, default_value):
    """Add a new parameter to SimParams __init__."""
    # Find the last parameter in __init__
    pattern = r"(self\.raid_conquest_prob\s*=\s*raid_conquest_prob)"
    if param_name in content:
        return content  # already added
    replacement = f"\\1\n        self.{param_name} = {param_name}"
    content = re.sub(pattern, replacement, content)
    # Add to __init__ signature
    init_pattern = r"(raid_conquest_prob=0\.3,?\s*\))"
    content = re.sub(init_pattern, f"raid_conquest_prob=0.3,\n                 {param_name}={default_value}):", content)
    return content


def implement_food_sharing(content):
    """H1: Allied neighbors share food — average food among allies."""
    print("  Implementing: food sharing between allied neighbors")

    # Add param
    if "food_sharing_rate" not in content:
        # Add after food_forest_bonus in SimParams
        content = content.replace(
            "self.food_forest_bonus = food_forest_bonus",
            "self.food_forest_bonus = food_forest_bonus\n"
            "        self.food_sharing_rate = food_sharing_rate"
        )
        content = content.replace(
            "food_forest_bonus=0.04,",
            "food_forest_bonus=0.04,\n                 food_sharing_rate=0.0,"
        )

    # Add food sharing logic after food_delta calculation
    sharing_code = '''
        # Food sharing among allied neighbors (H1)
        if p.food_sharing_rate > 0 and n > 1:
            new_food = self.s_food.copy()
            for i in range(n):
                allied_food = []
                for j in range(n):
                    if i != j and self.s_owner[i] == self.s_owner[j]:
                        dx = abs(int(self.s_x[i]) - int(self.s_x[j]))
                        dy = abs(int(self.s_y[i]) - int(self.s_y[j]))
                        if dx + dy <= 2:
                            allied_food.append(self.s_food[j])
                if allied_food:
                    avg_allied = np.mean(allied_food)
                    new_food[i] += p.food_sharing_rate * (avg_allied - self.s_food[i])
            self.s_food = np.clip(new_food, 0.0, 1.0)
'''
    marker = "        self.s_food = np.clip(self.s_food + food_delta, 0.0, 1.0)"
    if "Food sharing among allied" not in content:
        content = content.replace(marker, marker + sharing_code)

    return content


def implement_faction_territory(content):
    """H5: Allied neighbors reduce death probability."""
    print("  Implementing: faction territory effect on death")

    # Add param
    if "allied_death_reduction" not in content:
        content = content.replace(
            "self.death_competition_factor = death_competition_factor",
            "self.death_competition_factor = death_competition_factor\n"
            "        self.allied_death_reduction = allied_death_reduction"
        )
        content = content.replace(
            "death_competition_factor=0.01,",
            "death_competition_factor=0.01,\n                 allied_death_reduction=0.0,"
        )

    # Add allied count to death calculation
    territory_code = '''
        # Faction territory: allied neighbors reduce death (H5)
        if p.allied_death_reduction > 0 and n > 1:
            allied_count = np.zeros(n)
            for i in range(n):
                for j in range(n):
                    if i != j and self.s_owner[i] == self.s_owner[j]:
                        dx = abs(int(self.s_x[i]) - int(self.s_x[j]))
                        dy = abs(int(self.s_y[i]) - int(self.s_y[j]))
                        if dx + dy <= 2:
                            allied_count[i] += 1
            death_prob -= p.allied_death_reduction * allied_count
'''
    marker = "        death_prob = np.clip(death_prob, 0.01, 0.5)"
    if "Faction territory" not in content:
        content = content.replace(marker, territory_code + "\n" + marker)

    return content


def implement_port_bays(content):
    """H6: Ports prefer bay locations (more ocean neighbors)."""
    print("  Implementing: port bay preference")

    if "port_bay_bonus" not in content:
        content = content.replace(
            "self.port_survival_bonus = port_survival_bonus",
            "self.port_survival_bonus = port_survival_bonus\n"
            "        self.port_bay_bonus = port_bay_bonus"
        )
        content = content.replace(
            "port_survival_bonus=0.0,",
            "port_survival_bonus=0.0,\n                 port_bay_bonus=0.0,"
        )

    return content


def implement_temporal_spawn(content):
    """H3: Spawn probability increases over time."""
    print("  Implementing: temporal spawn rate increase")

    if "spawn_temporal_growth" not in content:
        content = content.replace(
            "self.spawn_prob = spawn_prob",
            "self.spawn_prob = spawn_prob\n"
            "        self.spawn_temporal_growth = spawn_temporal_growth"
        )
        content = content.replace(
            "spawn_prob=0.10,",
            "spawn_prob=0.10,\n                 spawn_temporal_growth=0.0,"
        )

    # Modify spawn probability to scale with time
    if "spawn_temporal_growth" not in content.split("_spawn_settlements")[1][:500]:
        old = "spawn_prob = p.spawn_prob"
        new = ("spawn_prob = p.spawn_prob * (1.0 + p.spawn_temporal_growth * "
               "self._step_count / 50.0)")
        content = content.replace(old, new, 1)

    return content


def main():
    results = load_results()
    exp_path = "simulator_exp.py"

    if not os.path.exists(exp_path):
        print(f"ERROR: {exp_path} not found. Run 'cp simulator.py simulator_exp.py' first.")
        sys.exit(1)

    content = read_file(exp_path)
    original = content

    confirmed = []
    for key, result in results.items():
        if result.get("confirmed"):
            confirmed.append(key)

    print(f"Confirmed hypotheses: {confirmed}")
    if not confirmed:
        print("No confirmed hypotheses. Nothing to implement.")
        return

    for key in confirmed:
        if key == "h1_food_sharing":
            content = implement_food_sharing(content)
        elif key == "h3_temporal_spawn":
            content = implement_temporal_spawn(content)
        elif key == "h5_faction_territory":
            content = implement_faction_territory(content)
        elif key == "h6_port_bays":
            content = implement_port_bays(content)
        else:
            print(f"  Skipping {key} — no implementation defined")

    if content != original:
        write_file(exp_path, content)
        print(f"\nPatched {exp_path}")
        # Syntax check
        try:
            compile(content, exp_path, "exec")
            print("Syntax check: OK")
        except SyntaxError as e:
            print(f"SYNTAX ERROR: {e}")
            write_file(exp_path, original)
            print("Reverted to original")
    else:
        print("No changes made")


if __name__ == "__main__":
    main()
