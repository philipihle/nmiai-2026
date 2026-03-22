"""Test 6 hypotheses about simulator mechanics against replay data.

Read-only analysis — does NOT modify any simulator files.
Outputs hypothesis_results.json with statistical tests.

Usage:
  python test_hypotheses.py                # test all, R1-R17
  python test_hypotheses.py --rounds 1 5 9 # specific rounds
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

CACHE_DIR = "replay_cache"


def load_replays(rounds=None):
    """Load all cached replay data."""
    replays = []
    for rn in range(1, 30):
        if rounds and rn not in rounds:
            continue
        detail_path = os.path.join(CACHE_DIR, f"round{rn}_detail.json")
        if not os.path.exists(detail_path):
            continue
        with open(detail_path) as f:
            detail = json.load(f)
        for si in range(detail.get("seeds_count", 5)):
            replay_path = os.path.join(CACHE_DIR, f"round{rn}_seed{si}_replay.json")
            if not os.path.exists(replay_path):
                continue
            with open(replay_path) as f:
                replay = json.load(f)
            replays.append({
                "round": rn,
                "seed": si,
                "grid": np.array(detail["initial_states"][si]["grid"]),
                "frames": replay["frames"],
            })
    return replays


def settlements_from_frame(frame):
    """Extract settlement positions and attributes from a frame."""
    slist = []
    for s in frame.get("settlements", []):
        slist.append({
            "x": s["x"], "y": s["y"],
            "pop": s.get("population", 0.5),
            "food": s.get("food", 0.5),
            "wealth": s.get("wealth", 0),
            "defense": s.get("defense", 0.2),
            "has_port": s.get("has_port", False),
            "owner_id": s.get("owner_id", 0),
        })
    return slist


def settle_set(settlements):
    return {(s["x"], s["y"]) for s in settlements}


def neighbors(x, y, radius=2):
    """Manhattan neighbors within radius."""
    result = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if abs(dx) + abs(dy) <= radius and (dx, dy) != (0, 0):
                result.append((x + dx, y + dy))
    return result


# ---- HYPOTHESIS 1: Food-sharing between allied neighbors ----
def test_food_sharing(replays):
    """Test if settlements with allied neighbors have higher food."""
    allied_food_pairs = []  # (n_allied, food_next_step)

    for rep in replays:
        frames = rep["frames"]
        for t in range(len(frames) - 1):
            curr = settlements_from_frame(frames[t])
            next_s = settlements_from_frame(frames[t + 1])
            if len(curr) < 3:
                continue

            pos_to_settle = {(s["x"], s["y"]): s for s in curr}
            next_pos = {(s["x"], s["y"]): s for s in next_s}

            for s in curr:
                key = (s["x"], s["y"])
                if key not in next_pos:
                    continue
                # Count allied neighbors
                n_allied = 0
                for nx, ny in neighbors(s["x"], s["y"], radius=2):
                    nb = pos_to_settle.get((nx, ny))
                    if nb and nb["owner_id"] == s["owner_id"]:
                        n_allied += 1
                allied_food_pairs.append((n_allied, next_pos[key]["food"]))

    if len(allied_food_pairs) < 100:
        return {"confirmed": False, "reason": "insufficient data", "n": len(allied_food_pairs)}

    data = np.array(allied_food_pairs)
    r, p = stats.pearsonr(data[:, 0], data[:, 1])
    return {
        "confirmed": abs(r) > 0.3 and p < 0.01,
        "r": float(r), "p": float(p), "n": len(data),
        "description": "Food-sharing between allied neighbors",
    }


# ---- HYPOTHESIS 2: Spawn direction bias ----
def test_spawn_direction(replays):
    """Test if spawns are biased toward forest / away from ocean."""
    terrain_at_spawn = {"forest": 0, "ocean_adj": 0, "other": 0, "total": 0}
    direction_counts = defaultdict(int)

    for rep in replays:
        grid = rep["grid"]
        H, W = grid.shape
        frames = rep["frames"]
        for t in range(len(frames) - 1):
            curr_set = settle_set(settlements_from_frame(frames[t]))
            next_set = settle_set(settlements_from_frame(frames[t + 1]))
            new_spawns = next_set - curr_set

            for (sx, sy) in new_spawns:
                if sx < 0 or sx >= W or sy < 0 or sy >= H:
                    continue
                terrain_at_spawn["total"] += 1
                init_terrain = grid[sy, sx]
                if init_terrain == 4:
                    terrain_at_spawn["forest"] += 1
                elif init_terrain == 10:
                    terrain_at_spawn["ocean_adj"] += 1
                else:
                    terrain_at_spawn["other"] += 1

                # Find nearest parent
                min_dist = 999
                parent = None
                for (px, py) in curr_set:
                    d = abs(px - sx) + abs(py - sy)
                    if 0 < d < min_dist:
                        min_dist = d
                        parent = (px, py)

                if parent:
                    dx = sx - parent[0]
                    dy = sy - parent[1]
                    # Classify direction
                    if abs(dx) >= abs(dy):
                        direction_counts["E" if dx > 0 else "W"] += 1
                    else:
                        direction_counts["S" if dy > 0 else "N"] += 1

    total = terrain_at_spawn["total"]
    if total < 100:
        return {"confirmed": False, "reason": "insufficient data", "n": total}

    forest_ratio = terrain_at_spawn["forest"] / total if total else 0
    # Expected forest ratio from initial map composition
    expected_forest = 0.20  # approximate

    bias = forest_ratio / expected_forest if expected_forest > 0 else 1.0
    # Chi-squared on direction
    dir_vals = list(direction_counts.values())
    if len(dir_vals) >= 2:
        chi2, p_dir = stats.chisquare(dir_vals)
    else:
        chi2, p_dir = 0, 1.0

    return {
        "confirmed": bias > 1.5,
        "forest_ratio": float(forest_ratio),
        "expected_forest": expected_forest,
        "bias": float(bias),
        "direction_chi2": float(chi2), "direction_p": float(p_dir),
        "directions": dict(direction_counts),
        "n": total,
        "description": "Spawn direction bias toward forest",
    }


# ---- HYPOTHESIS 3: Temporal spawn-rate increase ----
def test_temporal_spawn(replays):
    """Test if spawn rate increases over time."""
    step_spawns = defaultdict(list)  # step -> [n_spawns]

    for rep in replays:
        frames = rep["frames"]
        for t in range(len(frames) - 1):
            curr = settlements_from_frame(frames[t])
            next_s = settlements_from_frame(frames[t + 1])
            curr_set = settle_set(curr)
            next_set = settle_set(next_s)

            n_spawns = len(next_set - curr_set)
            n_eligible = max(1, len(curr))
            step_spawns[t].append(n_spawns / n_eligible)

    if len(step_spawns) < 10:
        return {"confirmed": False, "reason": "insufficient data"}

    steps = sorted(step_spawns.keys())
    avg_rates = [np.mean(step_spawns[s]) for s in steps]

    slope, intercept, r, p, se = stats.linregress(steps, avg_rates)
    r2 = r ** 2

    return {
        "confirmed": r2 > 0.3 and slope > 0,
        "slope": float(slope), "intercept": float(intercept),
        "r2": float(r2), "p": float(p),
        "n_steps": len(steps),
        "description": "Temporal spawn-rate increase",
    }


# ---- HYPOTHESIS 4: Parent attribute inheritance ----
def test_parent_inheritance(replays):
    """Test if spawned settlements inherit parent attributes."""
    parent_child_pairs = []  # (parent_pop, parent_food, child_pop, child_food)

    for rep in replays:
        frames = rep["frames"]
        for t in range(len(frames) - 1):
            curr = settlements_from_frame(frames[t])
            next_s = settlements_from_frame(frames[t + 1])
            curr_pos = {(s["x"], s["y"]): s for s in curr}
            next_pos = {(s["x"], s["y"]): s for s in next_s}
            curr_set = set(curr_pos.keys())
            next_set = set(next_pos.keys())

            for (sx, sy) in next_set - curr_set:
                child = next_pos[(sx, sy)]
                # Find nearest parent
                min_dist = 999
                parent = None
                for (px, py) in curr_set:
                    d = abs(px - sx) + abs(py - sy)
                    if 0 < d <= 3 and d < min_dist:
                        min_dist = d
                        parent = curr_pos[(px, py)]

                if parent:
                    parent_child_pairs.append((
                        parent["pop"], parent["food"],
                        child["pop"], child["food"],
                    ))

    if len(parent_child_pairs) < 100:
        return {"confirmed": False, "reason": "insufficient data", "n": len(parent_child_pairs)}

    data = np.array(parent_child_pairs)
    r_pop, p_pop = stats.pearsonr(data[:, 0], data[:, 2])
    r_food, p_food = stats.pearsonr(data[:, 1], data[:, 3])

    return {
        "confirmed": abs(r_pop) > 0.3 or abs(r_food) > 0.3,
        "r_pop": float(r_pop), "p_pop": float(p_pop),
        "r_food": float(r_food), "p_food": float(p_food),
        "n": len(data),
        "description": "Parent attribute inheritance to spawned settlements",
    }


# ---- HYPOTHESIS 5: Faction territory effect ----
def test_faction_territory(replays):
    """Test if allied neighbors reduce death probability."""
    alive_data = []  # (n_allied, n_enemy, survived)

    for rep in replays:
        frames = rep["frames"]
        for t in range(len(frames) - 1):
            curr = settlements_from_frame(frames[t])
            next_s = settlements_from_frame(frames[t + 1])
            if len(curr) < 3:
                continue

            curr_pos = {(s["x"], s["y"]): s for s in curr}
            next_alive = settle_set(next_s)

            for s in curr:
                key = (s["x"], s["y"])
                n_allied = 0
                n_enemy = 0
                for nx, ny in neighbors(s["x"], s["y"], radius=2):
                    nb = curr_pos.get((nx, ny))
                    if nb:
                        if nb["owner_id"] == s["owner_id"]:
                            n_allied += 1
                        else:
                            n_enemy += 1
                survived = 1 if key in next_alive else 0
                alive_data.append((n_allied, n_enemy, survived))

    if len(alive_data) < 200:
        return {"confirmed": False, "reason": "insufficient data", "n": len(alive_data)}

    data = np.array(alive_data)
    # Logistic-like: survival rate by allied count
    survival_by_allied = {}
    for n_a in range(int(data[:, 0].max()) + 1):
        mask = data[:, 0] == n_a
        if mask.sum() > 10:
            survival_by_allied[int(n_a)] = float(data[mask, 2].mean())

    # Simple correlation
    r_allied, p_allied = stats.pearsonr(data[:, 0], data[:, 2])
    r_enemy, p_enemy = stats.pearsonr(data[:, 1], data[:, 2])

    return {
        "confirmed": r_allied > 0.05 and p_allied < 0.01,
        "r_allied": float(r_allied), "p_allied": float(p_allied),
        "r_enemy": float(r_enemy), "p_enemy": float(p_enemy),
        "survival_by_allied": survival_by_allied,
        "n": len(data),
        "description": "Allied neighbors reduce death probability",
    }


# ---- HYPOTHESIS 6: Port placement preference (bays) ----
def test_port_bays(replays):
    """Test if ports prefer locations with more ocean neighbors."""
    port_ocean_adj = []
    non_port_ocean_adj = []

    for rep in replays:
        grid = rep["grid"]
        H, W = grid.shape
        frames = rep["frames"]

        # Build ocean adjacency map
        ocean = grid == 10
        ocean_adj = np.zeros((H, W), dtype=int)
        if H > 1:
            ocean_adj[1:] += ocean[:-1]
            ocean_adj[:-1] += ocean[1:]
        if W > 1:
            ocean_adj[:, 1:] += ocean[:, :-1]
            ocean_adj[:, :-1] += ocean[:, 1:]

        for t in range(len(frames) - 1):
            curr = settlements_from_frame(frames[t])
            next_s = settlements_from_frame(frames[t + 1])
            curr_ports = {(s["x"], s["y"]) for s in curr if s["has_port"]}
            next_ports = {(s["x"], s["y"]) for s in next_s if s["has_port"]}

            new_ports = next_ports - curr_ports
            for (px, py) in new_ports:
                if 0 <= px < W and 0 <= py < H:
                    port_ocean_adj.append(ocean_adj[py, px])

            # Non-port coastal settlements
            for s in next_s:
                key = (s["x"], s["y"])
                if not s["has_port"] and 0 <= s["x"] < W and 0 <= s["y"] < H:
                    oa = ocean_adj[s["y"], s["x"]]
                    if oa > 0:  # coastal
                        non_port_ocean_adj.append(oa)

    if len(port_ocean_adj) < 20 or len(non_port_ocean_adj) < 20:
        return {"confirmed": False, "reason": "insufficient data",
                "n_ports": len(port_ocean_adj), "n_non_ports": len(non_port_ocean_adj)}

    stat, p = stats.ks_2samp(port_ocean_adj, non_port_ocean_adj)
    port_mean = np.mean(port_ocean_adj)
    non_port_mean = np.mean(non_port_ocean_adj)

    return {
        "confirmed": p < 0.01 and port_mean > non_port_mean,
        "ks_stat": float(stat), "p": float(p),
        "port_mean_ocean_adj": float(port_mean),
        "non_port_mean_ocean_adj": float(non_port_mean),
        "n_ports": len(port_ocean_adj),
        "n_non_ports": len(non_port_ocean_adj),
        "description": "Ports prefer bay locations (more ocean neighbors)",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    print("Loading replay data...")
    replays = load_replays(args.rounds)
    print(f"Loaded {len(replays)} replays")

    if not replays:
        print("No replay data found!")
        sys.exit(1)

    results = {}

    tests = [
        ("h1_food_sharing", test_food_sharing),
        ("h2_spawn_direction", test_spawn_direction),
        ("h3_temporal_spawn", test_temporal_spawn),
        ("h4_parent_inheritance", test_parent_inheritance),
        ("h5_faction_territory", test_faction_territory),
        ("h6_port_bays", test_port_bays),
    ]

    for name, func in tests:
        print(f"\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        try:
            result = func(replays)
            results[name] = result
            status = "CONFIRMED" if result["confirmed"] else "NOT CONFIRMED"
            print(f"Result: {status}")
            for k, v in result.items():
                if k not in ("confirmed", "description"):
                    print(f"  {k}: {v}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[name] = {"confirmed": False, "error": str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    confirmed = [k for k, v in results.items() if v.get("confirmed")]
    not_confirmed = [k for k, v in results.items() if not v.get("confirmed")]
    print(f"Confirmed ({len(confirmed)}): {confirmed}")
    print(f"Not confirmed ({len(not_confirmed)}): {not_confirmed}")

    with open("hypothesis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to hypothesis_results.json")


if __name__ == "__main__":
    main()
