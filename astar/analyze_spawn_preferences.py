#!/usr/bin/env python3
"""
Analyze spawn preferences from replay data (R1-R21).

For each spawn event, records features of the chosen cell AND all eligible
non-chosen cells near the parent, then compares to learn preference weights.
"""

import json
import glob
import numpy as np
from collections import defaultdict

GRID_SIZE = 40
TERRAIN_NAMES = {10: "ocean", 5: "mountain", 11: "empty", 4: "forest", 3: "ruin", 1: "settlement", 2: "port"}
SPAWNABLE_TERRAIN = {11, 4, 3}  # empty, forest, ruin
MAX_PARENT_DIST = 3  # chebyshev distance


def chebyshev(x1, y1, x2, y2):
    return max(abs(x1 - x2), abs(y1 - y2))


def manhattan(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)


def get_neighbors(x, y, d=1):
    """Get all cells within chebyshev distance d."""
    cells = []
    for dx in range(-d, d + 1):
        for dy in range(-d, d + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                cells.append((nx, ny))
    return cells


def distance_to_nearest_ocean(grid, x, y):
    """BFS to find nearest ocean cell."""
    from collections import deque
    visited = set()
    queue = deque([(x, y, 0)])
    visited.add((x, y))
    while queue:
        cx, cy, dist = queue.popleft()
        if grid[cy][cx] == 10:
            return dist
        if dist > 15:  # cap search
            return 99
        for nx, ny in get_neighbors(cx, cy, d=1):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny, dist + 1))
    return 99


def extract_features(grid, x, y):
    """Extract features for a cell given the grid state BEFORE spawning."""
    terrain = grid[y][x]

    # Count settlement neighbors within d<=2
    settlement_neighbors = 0
    for nx, ny in get_neighbors(x, y, d=2):
        if grid[ny][nx] in (1, 2):  # settlement or port
            settlement_neighbors += 1

    # Count forest-adjacent cells (d=1 neighbors with forest)
    forest_adjacent = 0
    for nx, ny in get_neighbors(x, y, d=1):
        if grid[ny][nx] == 4:
            forest_adjacent += 1

    # Coastal: any d=1 neighbor is ocean
    coastal = 0
    for nx, ny in get_neighbors(x, y, d=1):
        if grid[ny][nx] == 10:
            coastal = 1
            break

    # Distance to nearest ocean
    ocean_dist = distance_to_nearest_ocean(grid, x, y)

    # Count mountain neighbors (d=1)
    mountain_adjacent = 0
    for nx, ny in get_neighbors(x, y, d=1):
        if grid[ny][nx] == 5:
            mountain_adjacent += 1

    # Count empty neighbors (d=1) - open space
    empty_adjacent = 0
    for nx, ny in get_neighbors(x, y, d=1):
        if grid[ny][nx] == 11:
            empty_adjacent += 1

    return {
        "terrain": terrain,
        "terrain_name": TERRAIN_NAMES.get(terrain, f"unknown({terrain})"),
        "settlement_neighbors_d2": settlement_neighbors,
        "forest_adjacent": forest_adjacent,
        "coastal": coastal,
        "ocean_dist": ocean_dist,
        "mountain_adjacent": mountain_adjacent,
        "empty_adjacent": empty_adjacent,
    }


def analyze_replays():
    chosen_features = []
    nonchosen_features = []
    spawn_count = 0
    skip_no_parent = 0
    skip_ambiguous = 0

    replay_files = sorted(glob.glob("replay_cache/round*_seed*_replay.json"))
    print(f"Found {len(replay_files)} replay files")

    for fpath in replay_files:
        with open(fpath) as f:
            data = json.load(f)

        round_id = data.get("round_id", "?")
        seed_idx = data.get("seed_index", "?")
        frames = data["frames"]

        for fi in range(len(frames) - 1):
            f0 = frames[fi]
            f1 = frames[fi + 1]
            grid = f0["grid"]  # grid BEFORE spawn

            # Find alive settlements in each frame
            alive0 = {(s["x"], s["y"]): s for s in f0["settlements"] if s["alive"]}
            alive1 = {(s["x"], s["y"]): s for s in f1["settlements"] if s["alive"]}

            # New settlement positions
            new_positions = set(alive1.keys()) - set(alive0.keys())

            for pos in new_positions:
                new_s = alive1[pos]
                owner = new_s["owner_id"]
                sx, sy = pos

                # Find parent: same owner_id, alive in f0, closest chebyshev
                parent_candidates = []
                for (px, py), ps in alive0.items():
                    if ps["owner_id"] == owner:
                        d = chebyshev(sx, sy, px, py)
                        parent_candidates.append((d, px, py))

                if not parent_candidates:
                    skip_no_parent += 1
                    continue

                parent_candidates.sort()
                best_dist = parent_candidates[0][0]

                if best_dist > MAX_PARENT_DIST:
                    skip_no_parent += 1
                    continue

                # Use closest parent
                _, parent_x, parent_y = parent_candidates[0]

                spawn_count += 1

                # Extract features for the chosen cell
                chosen_feat = extract_features(grid, sx, sy)
                chosen_feat["dist_to_parent"] = best_dist
                chosen_features.append(chosen_feat)

                # Find all eligible non-chosen cells within d<=3 of parent
                occupied = set(alive0.keys())
                for nx, ny in get_neighbors(parent_x, parent_y, d=MAX_PARENT_DIST):
                    if (nx, ny) == (sx, sy):
                        continue  # skip the chosen cell
                    if (nx, ny) in occupied:
                        continue  # already has settlement
                    terrain_val = grid[ny][nx]
                    if terrain_val not in SPAWNABLE_TERRAIN:
                        continue  # not spawnable terrain

                    nc_feat = extract_features(grid, nx, ny)
                    nc_feat["dist_to_parent"] = chebyshev(parent_x, parent_y, nx, ny)
                    nonchosen_features.append(nc_feat)

    print(f"\nSpawn events analyzed: {spawn_count}")
    print(f"Skipped (no parent / parent too far): {skip_no_parent}")
    print(f"Chosen cells recorded: {len(chosen_features)}")
    print(f"Non-chosen cells recorded: {len(nonchosen_features)}")

    return chosen_features, nonchosen_features


def compute_stats(features_list):
    """Compute mean and std for each numeric feature."""
    if not features_list:
        return {}
    keys = [k for k in features_list[0] if k not in ("terrain", "terrain_name")]
    stats = {}
    for k in keys:
        vals = [f[k] for f in features_list]
        stats[k] = {"mean": np.mean(vals), "std": np.std(vals), "median": np.median(vals)}
    return stats


def compute_terrain_distribution(features_list):
    """Count terrain type distribution."""
    counts = defaultdict(int)
    for f in features_list:
        counts[f["terrain_name"]] += 1
    total = len(features_list)
    return {k: (v, v / total * 100) for k, v in sorted(counts.items(), key=lambda x: -x[1])}


def logistic_regression_weights(chosen, nonchosen):
    """Simple logistic regression to learn preference weights."""
    feature_keys = ["settlement_neighbors_d2", "forest_adjacent", "coastal",
                    "ocean_dist", "mountain_adjacent", "empty_adjacent", "dist_to_parent"]

    # Add terrain one-hot features
    terrain_types = ["empty", "forest", "ruin"]

    # Build feature matrix
    X = []
    y = []

    for f in chosen:
        row = [f[k] for k in feature_keys]
        for t in terrain_types:
            row.append(1.0 if f["terrain_name"] == t else 0.0)
        X.append(row)
        y.append(1)

    for f in nonchosen:
        row = [f[k] for k in feature_keys]
        for t in terrain_types:
            row.append(1.0 if f["terrain_name"] == t else 0.0)
        X.append(row)
        y.append(0)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    all_names = feature_keys + [f"terrain_{t}" for t in terrain_types]

    # Normalize features
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1  # avoid division by zero
    X_norm = (X - means) / stds

    # Add intercept
    X_norm = np.column_stack([np.ones(len(X_norm)), X_norm])

    # Gradient descent logistic regression
    n_features = X_norm.shape[1]
    weights = np.zeros(n_features)
    lr = 0.01
    n_iter = 5000

    for _ in range(n_iter):
        logits = X_norm @ weights
        logits = np.clip(logits, -20, 20)
        probs = 1 / (1 + np.exp(-logits))
        grad = X_norm.T @ (probs - y) / len(y)
        # L2 regularization
        grad[1:] += 0.001 * weights[1:]
        weights -= lr * grad

    # Final accuracy
    preds = (X_norm @ weights > 0).astype(float)
    accuracy = np.mean(preds == y)

    return weights, all_names, means, stds, accuracy


def main():
    import os
    os.chdir(os.path.expanduser("~/astar-island"))

    print("=" * 70)
    print("SPAWN PREFERENCE ANALYSIS - R1 to R21")
    print("=" * 70)

    chosen, nonchosen = analyze_replays()

    if not chosen:
        print("No spawn events found!")
        return

    # --- Feature comparison ---
    print("\n" + "=" * 70)
    print("FEATURE COMPARISON: CHOSEN vs NON-CHOSEN CELLS")
    print("=" * 70)

    chosen_stats = compute_stats(chosen)
    nonchosen_stats = compute_stats(nonchosen)

    print(f"\n{'Feature':<28} {'Chosen Mean':>12} {'NonChosen Mean':>14} {'Diff':>8} {'Signal':>8}")
    print("-" * 72)
    for key in chosen_stats:
        cm = chosen_stats[key]["mean"]
        nm = nonchosen_stats[key]["mean"]
        diff = cm - nm
        # Effect size (Cohen's d approximation)
        pooled_std = np.sqrt((chosen_stats[key]["std"]**2 + nonchosen_stats[key]["std"]**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        signal = "***" if abs(cohens_d) > 0.5 else "**" if abs(cohens_d) > 0.3 else "*" if abs(cohens_d) > 0.1 else ""
        print(f"{key:<28} {cm:>12.3f} {nm:>14.3f} {diff:>+8.3f} {signal:>8}")

    # --- Terrain distribution ---
    print("\n" + "=" * 70)
    print("TERRAIN DISTRIBUTION")
    print("=" * 70)

    chosen_terrain = compute_terrain_distribution(chosen)
    nonchosen_terrain = compute_terrain_distribution(nonchosen)

    print(f"\n{'Terrain':<12} {'Chosen':>12} {'Chosen%':>8} {'NonChosen':>12} {'NonCh%':>8} {'Pref Ratio':>11}")
    print("-" * 65)
    all_terrains = set(list(chosen_terrain.keys()) + list(nonchosen_terrain.keys()))
    for t in sorted(all_terrains):
        cc, cp = chosen_terrain.get(t, (0, 0))
        nc, np_ = nonchosen_terrain.get(t, (0, 0))
        ratio = (cp / np_) if np_ > 0 else float("inf")
        print(f"{t:<12} {cc:>12} {cp:>7.1f}% {nc:>12} {np_:>7.1f}% {ratio:>10.2f}x")

    # --- Distance to parent ---
    print("\n" + "=" * 70)
    print("DISTANCE TO PARENT DISTRIBUTION (Chebyshev)")
    print("=" * 70)
    dist_counts = defaultdict(int)
    for f in chosen:
        dist_counts[f["dist_to_parent"]] += 1
    total = len(chosen)
    for d in sorted(dist_counts.keys()):
        c = dist_counts[d]
        bar = "#" * int(c / total * 60)
        print(f"  d={d}: {c:>6} ({c/total*100:>5.1f}%) {bar}")

    # --- Logistic Regression ---
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION PREFERENCE WEIGHTS")
    print("=" * 70)

    weights, names, means, stds, accuracy = logistic_regression_weights(chosen, nonchosen)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Intercept: {weights[0]:+.4f}")
    print(f"\n{'Feature':<28} {'Weight':>10} {'Direction':>12}")
    print("-" * 52)

    # Sort by absolute weight
    indexed = list(enumerate(names))
    indexed.sort(key=lambda x: abs(weights[x[0] + 1]), reverse=True)

    for idx, name in indexed:
        w = weights[idx + 1]
        direction = "PREFER" if w > 0 else "AVOID"
        print(f"{name:<28} {w:>+10.4f} {direction:>12}")

    # --- Summary preference table ---
    print("\n" + "=" * 70)
    print("SUMMARY: SPAWN PREFERENCE WEIGHT TABLE")
    print("=" * 70)
    print("\nHigher weight = stronger preference for spawning on that feature.")
    print("Weights are from standardized logistic regression.\n")

    # Raw preference ratios for terrain
    print("TERRAIN PREFERENCES (by selection ratio):")
    for t in sorted(all_terrains):
        cc, cp = chosen_terrain.get(t, (0, 0))
        nc, np_ = nonchosen_terrain.get(t, (0, 0))
        ratio = (cp / np_) if np_ > 0 else float("inf")
        pref = "STRONGLY PREFERRED" if ratio > 1.5 else "PREFERRED" if ratio > 1.1 else "NEUTRAL" if ratio > 0.9 else "AVOIDED"
        print(f"  {t:<12}: {ratio:.2f}x  ({pref})")

    print("\nCONTINUOUS FEATURE PREFERENCES (by mean difference, chosen - nonchosen):")
    for key in ["settlement_neighbors_d2", "forest_adjacent", "coastal",
                 "ocean_dist", "mountain_adjacent", "empty_adjacent", "dist_to_parent"]:
        cm = chosen_stats[key]["mean"]
        nm = nonchosen_stats[key]["mean"]
        diff = cm - nm
        pooled_std = np.sqrt((chosen_stats[key]["std"]**2 + nonchosen_stats[key]["std"]**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        if abs(cohens_d) > 0.5:
            strength = "STRONG"
        elif abs(cohens_d) > 0.3:
            strength = "MODERATE"
        elif abs(cohens_d) > 0.1:
            strength = "WEAK"
        else:
            strength = "NEGLIGIBLE"
        direction = "higher" if diff > 0 else "lower"
        print(f"  {key:<28}: {direction} by {abs(diff):.3f} (d={cohens_d:+.3f}, {strength})")

    # --- Conditional analysis: terrain-specific preferences ---
    print("\n" + "=" * 70)
    print("CONDITIONAL ANALYSIS: Features by Terrain Type")
    print("=" * 70)

    for terrain_name in ["ruin", "forest", "empty"]:
        chosen_t = [f for f in chosen if f["terrain_name"] == terrain_name]
        nonchosen_t = [f for f in nonchosen if f["terrain_name"] == terrain_name]
        if not chosen_t or not nonchosen_t:
            continue
        print(f"\n--- {terrain_name.upper()} cells (chosen={len(chosen_t)}, nonchosen={len(nonchosen_t)}) ---")
        cs = compute_stats(chosen_t)
        ns = compute_stats(nonchosen_t)
        for key in ["settlement_neighbors_d2", "forest_adjacent", "coastal", "ocean_dist", "dist_to_parent"]:
            cm = cs[key]["mean"]
            nm = ns[key]["mean"]
            diff = cm - nm
            print(f"  {key:<28}: chosen={cm:.3f}, nonchosen={nm:.3f}, diff={diff:+.3f}")


if __name__ == "__main__":
    main()
