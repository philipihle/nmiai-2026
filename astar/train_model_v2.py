"""
Train enhanced LightGBM model v2 for Astar Island settlement prediction.

Directly predicts settlement probability distributions from cell features,
WITHOUT relying on MC simulation. Uses per-round settle_rate as a key feature
to make predictions round-adaptive.

Training data: R1-R22 (22 rounds x 5 seeds = 110 examples, 40x40 cells each)
Features: ~45 features including terrain, spatial, density, and round-level features
Model: 6 separate LightGBM regressors (one per terrain class)
Evaluation: Leave-one-round-out cross-validation with competition scoring
"""

import numpy as np
import json
import pickle
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=UserWarning)

import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut

CACHE_DIR = os.path.expanduser("~/astar-island/replay_cache")
OUT_DIR = os.path.expanduser("~/astar-island")

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# Terrain codes in grid
TERRAIN_OCEAN = 10
TERRAIN_PLAINS = 11
TERRAIN_EMPTY = 0
TERRAIN_SETTLEMENT = 1
TERRAIN_PORT = 2
TERRAIN_RUIN = 3
TERRAIN_FOREST = 4
TERRAIN_MOUNTAIN = 5


def load_per_round_params():
    """Load settle_rate for each round from per_round_params.json."""
    path = os.path.join(CACHE_DIR, "per_round_params.json")
    with open(path) as f:
        data = json.load(f)
    return {entry["round"]: entry for entry in data}


def estimate_settle_rate_from_gt(gt, initial_grid):
    """Estimate settle_rate from ground truth when not available in params.
    
    settle_rate ~ average settlement probability on eligible (non-ocean, non-settlement) cells.
    """
    land_mask = initial_grid != TERRAIN_OCEAN
    not_settlement = ~np.isin(initial_grid, [TERRAIN_SETTLEMENT, TERRAIN_PORT])
    eligible = land_mask & not_settlement
    if eligible.sum() == 0:
        return 0.0
    settle_probs = gt[:, :, 1]  # settlement probability
    return float(settle_probs[eligible].mean())


def load_all_data():
    """Load all round data from cache files."""
    params = load_per_round_params()
    all_data = []

    for rn in range(1, 30):  # generous range
        detail_path = os.path.join(CACHE_DIR, f"round{rn}_detail.json")
        if not os.path.exists(detail_path):
            continue

        with open(detail_path) as f:
            detail = json.load(f)

        W = detail["map_width"]
        H = detail["map_height"]

        for si in range(detail["seeds_count"]):
            analysis_path = os.path.join(CACHE_DIR, f"round{rn}_seed{si}_analysis.json")
            if not os.path.exists(analysis_path):
                continue

            with open(analysis_path) as f:
                analysis = json.load(f)

            # Always use detail file for initial grid (analysis may not have it)
            initial_grid = np.array(detail["initial_states"][si]["grid"])
            gt = np.array(analysis["ground_truth"])
            settlements = detail["initial_states"][si]["settlements"]

            # Get settle_rate
            if rn in params:
                settle_rate = params[rn]["settle_rate"]
            else:
                settle_rate = estimate_settle_rate_from_gt(gt, initial_grid)
                print(f"  Round {rn}: estimated settle_rate={settle_rate:.6f}")

            all_data.append({
                "round_number": rn,
                "seed_index": si,
                "initial_grid": initial_grid,
                "ground_truth": gt,
                "settlements": settlements,
                "settle_rate": settle_rate,
                "W": W,
                "H": H,
            })

        n_seeds = sum(1 for d in all_data if d["round_number"] == rn)
        if n_seeds > 0:
            sr = all_data[-1]["settle_rate"]
            print(f"  Round {rn}: {n_seeds} seeds, settle_rate={sr:.4f}")

    print(f"\nTotal: {len(all_data)} examples from {len(set(d['round_number'] for d in all_data))} rounds")
    return all_data


def manhattan_dist_field(grid, target_mask):
    """Compute Manhattan distance from every cell to nearest cell in target_mask.

    Direct computation: for each cell, min Manhattan distance to any target cell.
    Fast enough for 40x40 grids.
    """
    H, W = grid.shape
    ys, xs = np.mgrid[0:H, 0:W]

    target_coords = np.argwhere(target_mask)  # (N, 2) array of [y, x]
    if len(target_coords) == 0:
        return np.full((H, W), 999.0)

    dist = np.full((H, W), 999.0)
    for ty, tx in target_coords:
        d = np.abs(ys - ty) + np.abs(xs - tx)
        dist = np.minimum(dist, d.astype(float))

    return dist


def compute_features(initial_grid, settlements_list, W, H, settle_rate):
    """Compute enhanced per-cell feature matrix.
    
    Returns (H*W, n_features) array and feature names.
    """
    grid = np.array(initial_grid) if not isinstance(initial_grid, np.ndarray) else initial_grid.copy()

    # Settlement positions
    spos = []
    for s in settlements_list:
        if s.get("alive", True):
            spos.append((s["x"], s["y"]))

    ys, xs = np.mgrid[0:H, 0:W]

    # --- TERRAIN ONE-HOT (7 features) ---
    is_ocean = (grid == TERRAIN_OCEAN).astype(float)
    is_plains = (grid == TERRAIN_PLAINS).astype(float)
    is_empty = (grid == TERRAIN_EMPTY).astype(float)
    is_forest = (grid == TERRAIN_FOREST).astype(float)
    is_mountain = (grid == TERRAIN_MOUNTAIN).astype(float)
    is_settlement = (grid == TERRAIN_SETTLEMENT).astype(float)
    is_port = (grid == TERRAIN_PORT).astype(float)

    # --- DISTANCE FEATURES (3 features) ---
    # Distance to nearest settlement
    dist_settle = np.full((H, W), 999.0)
    for sx, sy in spos:
        d = (np.abs(xs - sx) + np.abs(ys - sy)).astype(float)
        dist_settle = np.minimum(dist_settle, d)

    # Distance to 2nd nearest settlement
    dist_settle2 = np.full((H, W), 999.0)
    for sx, sy in spos:
        d = (np.abs(xs - sx) + np.abs(ys - sy)).astype(float)
        mask = d > dist_settle
        dist_settle2 = np.where(mask, np.minimum(dist_settle2, d), dist_settle2)

    # Distance to nearest ocean
    ocean_mask = grid == TERRAIN_OCEAN
    if ocean_mask.any():
        dist_ocean = manhattan_dist_field(grid, ocean_mask)
    else:
        dist_ocean = np.full((H, W), 999.0)

    # Distance to nearest forest
    forest_mask = grid == TERRAIN_FOREST
    if forest_mask.any():
        dist_forest = manhattan_dist_field(grid, forest_mask)
    else:
        dist_forest = np.full((H, W), 999.0)

    # --- COASTAL (1 feature) ---
    land = ~ocean_mask
    coastal = np.zeros((H, W), dtype=float)
    if H > 1:
        coastal[1:, :] += (land[1:, :] & ocean_mask[:-1, :]).astype(float)
        coastal[:-1, :] += (land[:-1, :] & ocean_mask[1:, :]).astype(float)
    if W > 1:
        coastal[:, 1:] += (land[:, 1:] & ocean_mask[:, :-1]).astype(float)
        coastal[:, :-1] += (land[:, :-1] & ocean_mask[:, 1:]).astype(float)

    # --- PENINSULA (1 feature) ---
    # True if >= 3 of 4 cardinal neighbors are ocean
    ocean_cardinal_count = np.zeros((H, W), dtype=float)
    if H > 1:
        ocean_cardinal_count[1:, :] += ocean_mask[:-1, :].astype(float)
        ocean_cardinal_count[:-1, :] += ocean_mask[1:, :].astype(float)
    if W > 1:
        ocean_cardinal_count[:, 1:] += ocean_mask[:, :-1].astype(float)
        ocean_cardinal_count[:, :-1] += ocean_mask[:, 1:].astype(float)
    is_peninsula = (ocean_cardinal_count >= 3).astype(float)

    # --- LOCAL DENSITY: Settlement counts at various radii (5 features) ---
    n_r1 = np.zeros((H, W), dtype=float)
    n_r2 = np.zeros((H, W), dtype=float)
    n_r3 = np.zeros((H, W), dtype=float)
    n_r5 = np.zeros((H, W), dtype=float)
    n_r8 = np.zeros((H, W), dtype=float)
    for sx, sy in spos:
        d = np.abs(xs - sx) + np.abs(ys - sy)
        n_r1 += (d <= 1).astype(float)
        n_r2 += (d <= 2).astype(float)
        n_r3 += (d <= 3).astype(float)
        n_r5 += (d <= 5).astype(float)
        n_r8 += (d <= 8).astype(float)

    # --- ADJACENT TERRAIN COUNTS (6 features) ---
    def count_adjacent(mask_val):
        m = (grid == mask_val).astype(float)
        c = np.zeros((H, W), dtype=float)
        if H > 1:
            c[1:] += m[:-1]
            c[:-1] += m[1:]
        if W > 1:
            c[:, 1:] += m[:, :-1]
            c[:, :-1] += m[:, 1:]
        return c

    adj_forest = count_adjacent(TERRAIN_FOREST)
    adj_ocean = count_adjacent(TERRAIN_OCEAN)
    adj_settlement = count_adjacent(TERRAIN_SETTLEMENT) + count_adjacent(TERRAIN_PORT)
    adj_plains = count_adjacent(TERRAIN_PLAINS)
    adj_mountain = count_adjacent(TERRAIN_MOUNTAIN)
    adj_empty = count_adjacent(TERRAIN_EMPTY)

    # --- TERRAIN PERCENTAGES in radius 3 (3 features) ---
    # Use rolling sum approach
    def terrain_pct_d3(mask_val):
        m = (grid == mask_val).astype(float)
        total = np.zeros((H, W), dtype=float)
        count = np.zeros((H, W), dtype=float)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if abs(dy) + abs(dx) <= 3:
                    y_shifted = np.clip(ys + dy, 0, H - 1)
                    x_shifted = np.clip(xs + dx, 0, W - 1)
                    total += m[y_shifted, x_shifted]
                    count += 1.0
        return total / count

    pct_forest_d3 = terrain_pct_d3(TERRAIN_FOREST)
    pct_ocean_d3 = terrain_pct_d3(TERRAIN_OCEAN)
    pct_settlement_d3 = terrain_pct_d3(TERRAIN_SETTLEMENT)

    # --- LAND CONNECTIVITY: land cells within radius 5 (1 feature) ---
    land_float = land.astype(float)
    land_r5 = np.zeros((H, W), dtype=float)
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            if abs(dy) + abs(dx) <= 5:
                y_shifted = np.clip(ys + dy, 0, H - 1)
                x_shifted = np.clip(xs + dx, 0, W - 1)
                land_r5 += land_float[y_shifted, x_shifted]

    # --- EDGE DISTANCE (1 feature) ---
    dist_edge = np.minimum(
        np.minimum(xs, W - 1 - xs),
        np.minimum(ys, H - 1 - ys)
    ).astype(float)

    # --- GLOBAL FEATURES (4 features) ---
    total_settlements = float(len(spos))
    total_land = float(np.sum(land))
    settlement_density = total_settlements / max(total_land, 1)

    # --- QUADRANT DENSITY (1 feature) ---
    mid_y, mid_x = H // 2, W // 2
    quad_counts = np.zeros(4)
    quad_areas = np.zeros(4)
    for sx, sy in spos:
        q = 0
        if sy >= mid_y:
            q += 2
        if sx >= mid_x:
            q += 1
        quad_counts[q] += 1
    # Areas
    quad_areas[0] = mid_y * mid_x
    quad_areas[1] = mid_y * (W - mid_x)
    quad_areas[2] = (H - mid_y) * mid_x
    quad_areas[3] = (H - mid_y) * (W - mid_x)

    quad_idx = (ys >= mid_y).astype(int) * 2 + (xs >= mid_x).astype(int)
    quad_densities = quad_counts / np.maximum(quad_areas, 1)
    quadrant_density = quad_densities[quad_idx]

    # --- POSITION FEATURES (2 features) ---
    # Normalized x, y position
    norm_x = xs.astype(float) / max(W - 1, 1)
    norm_y = ys.astype(float) / max(H - 1, 1)

    # --- RUIN DISTANCE (1 feature) ---
    ruin_mask = grid == TERRAIN_RUIN
    if ruin_mask.any():
        dist_ruin = manhattan_dist_field(grid, ruin_mask)
    else:
        dist_ruin = np.full((H, W), 999.0)

    # --- PORT FRACTION (1 feature) ---
    port_count = sum(1 for s in settlements_list if s.get("has_port", False) and s.get("alive", True))
    port_fraction = port_count / max(len(spos), 1)

    # --- SETTLE RATE (2 features: raw + log-transformed) ---
    settle_rate_arr = np.full((H, W), settle_rate)
    log_settle_rate = np.full((H, W), np.log(max(settle_rate, 1e-6)))

    # --- INTERACTION FEATURES (4 features) ---
    settle_rate_x_dist = settle_rate_arr * dist_settle
    settle_rate_x_density = settle_rate_arr * n_r3
    settle_rate_x_coastal = settle_rate_arr * coastal
    settle_rate_x_forest = settle_rate_arr * is_forest

    # --- Stack all features ---
    features = np.stack([
        # Terrain one-hot (0-6)
        is_ocean, is_plains, is_empty, is_forest, is_mountain, is_settlement, is_port,
        # Distances (7-11)
        dist_settle, dist_settle2, dist_ocean, dist_forest, dist_ruin,
        # Coastal/peninsula (12-13)
        coastal, is_peninsula,
        # Local density (14-18)
        n_r1, n_r2, n_r3, n_r5, n_r8,
        # Adjacent terrain (19-24)
        adj_forest, adj_ocean, adj_settlement, adj_plains, adj_mountain, adj_empty,
        # Terrain percentages d3 (25-27)
        pct_forest_d3, pct_ocean_d3, pct_settlement_d3,
        # Land connectivity (28)
        land_r5,
        # Edge distance (29)
        dist_edge,
        # Global (30-33)
        np.full((H, W), total_settlements),
        np.full((H, W), settlement_density),
        np.full((H, W), total_land),
        np.full((H, W), port_fraction),
        # Quadrant density (34)
        quadrant_density,
        # Position (35-36)
        norm_x, norm_y,
        # Settle rate (37-38)
        settle_rate_arr, log_settle_rate,
        # Interaction features (39-42)
        settle_rate_x_dist,
        settle_rate_x_density,
        settle_rate_x_coastal,
        settle_rate_x_forest,
        # Ocean cardinal count (43)
        ocean_cardinal_count,
    ], axis=-1)

    return features.reshape(H * W, -1)


FEATURE_NAMES = [
    # Terrain one-hot (0-6)
    "is_ocean", "is_plains", "is_empty", "is_forest", "is_mountain", "is_settlement", "is_port",
    # Distances (7-11)
    "dist_nearest_settle", "dist_2nd_settle", "dist_ocean", "dist_forest", "dist_ruin",
    # Coastal/peninsula (12-13)
    "coastal", "is_peninsula",
    # Local density (14-18)
    "n_settle_r1", "n_settle_r2", "n_settle_r3", "n_settle_r5", "n_settle_r8",
    # Adjacent terrain (19-24)
    "adj_forest", "adj_ocean", "adj_settlement", "adj_plains", "adj_mountain", "adj_empty",
    # Terrain percentages d3 (25-27)
    "pct_forest_d3", "pct_ocean_d3", "pct_settlement_d3",
    # Land connectivity (28)
    "land_r5",
    # Edge distance (29)
    "dist_edge",
    # Global (30-33)
    "total_settlements", "settlement_density", "total_land", "port_fraction",
    # Quadrant density (34)
    "quadrant_density",
    # Position (35-36)
    "norm_x", "norm_y",
    # Settle rate (37-38)
    "settle_rate", "log_settle_rate",
    # Interaction features (39-42)
    "settle_rate_x_dist",
    "settle_rate_x_density",
    "settle_rate_x_coastal",
    "settle_rate_x_forest",
    # Ocean cardinal count (43)
    "ocean_cardinal_count",
]


def build_dataset(all_data):
    """Build training dataset from all round data.

    Also computes sample weights: higher weight for cells with high entropy
    (the cells that matter for scoring), lower weight for ocean (trivial).
    """
    X_all = []
    y_all = []
    w_all = []
    groups = []  # round number for leave-one-round-out CV
    meta = []  # (round_number, seed_index) for each block of 1600

    for i, d in enumerate(all_data):
        grid = d["initial_grid"]
        gt = d["ground_truth"]
        W, H = d["W"], d["H"]
        settle_rate = d["settle_rate"]
        settlements = d["settlements"]

        features = compute_features(grid, settlements, W, H, settle_rate)
        targets = gt.reshape(H * W, -1)

        # Compute sample weights based on entropy (match scoring formula)
        eps = 1e-10
        entropy = -np.sum(targets * np.log(targets + eps), axis=-1)
        # Weight = 1 + 9 * normalized_entropy (ocean gets ~1, high-entropy gets ~10)
        max_ent = np.log(6)  # max possible entropy with 6 classes
        weights = 1.0 + 9.0 * (entropy / max_ent)

        # Sanity check dimensions
        assert features.shape[0] == H * W, f"Feature shape mismatch: {features.shape[0]} vs {H*W}"
        assert targets.shape == (H * W, 6), f"Target shape mismatch: {targets.shape}"

        X_all.append(features)
        y_all.append(targets)
        w_all.append(weights)
        groups.extend([d["round_number"]] * (H * W))
        meta.append((d["round_number"], d["seed_index"]))

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(all_data)} examples...")

    X = np.vstack(X_all)
    y = np.vstack(y_all)
    w = np.concatenate(w_all)
    groups = np.array(groups)

    print(f"\nDataset: {X.shape[0]} cells, {X.shape[1]} features, {y.shape[1]} classes")
    print(f"  Feature names count: {len(FEATURE_NAMES)}")
    print(f"  Weight stats: min={w.min():.1f}, max={w.max():.1f}, mean={w.mean():.1f}")
    assert X.shape[1] == len(FEATURE_NAMES), f"Feature count mismatch: {X.shape[1]} vs {len(FEATURE_NAMES)}"

    return X, y, w, groups, meta


def competition_score(y_true, y_pred):
    """Compute competition score: 100 * exp(-3 * entropy_weighted_kl)."""
    eps = 1e-10
    y_pred = np.maximum(y_pred, 0.001)
    y_pred = y_pred / y_pred.sum(axis=-1, keepdims=True)

    entropy = -np.sum(y_true * np.log(y_true + eps), axis=-1)
    kl = np.sum(y_true * np.log((y_true + eps) / (y_pred + eps)), axis=-1)

    total_entropy = entropy.sum()
    if total_entropy < eps:
        return 100.0

    weighted_kl = (entropy * kl).sum() / total_entropy
    return max(0, min(100, 100 * np.exp(-3 * weighted_kl)))


def train_cv(X, y, w, groups, meta):
    """Train LightGBM models with leave-one-round-out CV."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    n_classes = y.shape[1]
    unique_rounds = sorted(set(groups))
    n_cells = 1600  # 40*40

    print(f"\nLeave-one-round-out CV across {len(unique_rounds)} rounds:")
    print(f"{'='*70}")

    # LightGBM parameters
    lgb_params = {
        "n_estimators": 800,
        "max_depth": 8,
        "learning_rate": 0.02,
        "num_leaves": 127,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_samples": 20,
        "reg_alpha": 0.02,
        "reg_lambda": 0.05,
        "verbose": -1,
        "n_jobs": 4,
    }

    cv_scores = {}
    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(X, y, groups):
        test_round = groups[test_idx[0]]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = w[train_idx]

        # Train one model per class
        preds = np.zeros_like(y_test)
        for c in range(n_classes):
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(
                X_train, y_train[:, c],
                sample_weight=w_train,
                eval_set=[(X_test, y_test[:, c])],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
            )
            preds[:, c] = model.predict(X_test)

        # Clip and normalize
        preds = np.maximum(preds, 0.001)
        preds = preds / preds.sum(axis=-1, keepdims=True)

        # Score per seed
        n_test_seeds = len(test_idx) // n_cells
        seed_scores = []
        for s in range(n_test_seeds):
            start = s * n_cells
            end = (s + 1) * n_cells
            sc = competition_score(y_test[start:end], preds[start:end])
            seed_scores.append(sc)

        avg = np.mean(seed_scores)
        cv_scores[test_round] = avg
        seeds_str = ", ".join(f"{s:.1f}" for s in seed_scores)
        print(f"  Round {test_round:>2d}: avg={avg:5.1f}  seeds=[{seeds_str}]")

    overall_avg = np.mean(list(cv_scores.values()))
    print(f"{'='*70}")
    print(f"  CV AVERAGE: {overall_avg:.1f}")

    # Show best/worst
    sorted_scores = sorted(cv_scores.items(), key=lambda x: x[1])
    print(f"\n  Worst 5: {', '.join(f'R{r}={s:.1f}' for r, s in sorted_scores[:5])}")
    print(f"  Best 5:  {', '.join(f'R{r}={s:.1f}' for r, s in sorted_scores[-5:])}")

    return cv_scores


def train_final_model(X, y, w):
    """Train final model on all data and save."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    n_classes = y.shape[1]

    lgb_params = {
        "n_estimators": 800,
        "max_depth": 8,
        "learning_rate": 0.02,
        "num_leaves": 127,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "min_child_samples": 20,
        "reg_alpha": 0.02,
        "reg_lambda": 0.05,
        "verbose": -1,
        "n_jobs": 4,
    }

    models = []
    print("\nTraining final models on all data:")
    for c in range(n_classes):
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X, y[:, c], sample_weight=w)
        models.append(model)
        print(f"  {CLASS_NAMES[c]:>12s}: trained ({model.n_estimators} trees)")

    # Feature importance
    print(f"\n{'='*70}")
    print("Feature importance (top 10 per class):")
    print(f"{'='*70}")
    for c, model in enumerate(models):
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:10]
        items = [(FEATURE_NAMES[i], imp[i]) for i in top_idx]
        desc = ", ".join(f"{n}={v:.0f}" for n, v in items)
        print(f"  {CLASS_NAMES[c]:>12s}: {desc}")

    # Overall importance across all classes
    print(f"\n  {'Overall top 15':}")
    total_imp = np.zeros(len(FEATURE_NAMES))
    for model in models:
        imp = model.feature_importances_
        # Normalize per model
        total_imp += imp / max(imp.sum(), 1)
    top_idx = np.argsort(total_imp)[::-1][:15]
    for i, idx in enumerate(top_idx):
        print(f"    {i+1:2d}. {FEATURE_NAMES[idx]:25s} = {total_imp[idx]:.4f}")

    # Save models + feature names + metadata
    save_data = {
        "models": models,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "class_names": CLASS_NAMES,
        "version": 2,
    }
    save_path = os.path.join(OUT_DIR, "ml_models_v2.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)
    print(f"\nModels saved to {save_path}")

    return models


def compare_with_baselines(cv_scores):
    """Compare v2 model scores with MC-only baseline."""
    params = load_per_round_params()

    print(f"\n{'='*70}")
    print("Comparison with MC-only baseline:")
    print(f"{'='*70}")
    print(f"  {'Round':>6s} | {'MC-only':>8s} | {'ML v2':>8s} | {'Diff':>8s}")
    print(f"  {'-'*40}")

    mc_scores = {}
    for entry in params.values():
        rn = entry["round"] if isinstance(entry, dict) else entry
        if isinstance(entry, dict):
            mc_scores[entry["round"]] = entry["score"]

    diffs = []
    for rn in sorted(cv_scores.keys()):
        ml_score = cv_scores[rn]
        if rn in mc_scores:
            mc_score = mc_scores[rn]
            diff = ml_score - mc_score
            diffs.append(diff)
            marker = " **" if diff > 2 else (" !!" if diff < -2 else "")
            print(f"  R{rn:>4d} | {mc_score:8.1f} | {ml_score:8.1f} | {diff:+8.1f}{marker}")
        else:
            print(f"  R{rn:>4d} | {'N/A':>8s} | {ml_score:8.1f} |")

    if diffs:
        mc_avg = np.mean([mc_scores[rn] for rn in sorted(cv_scores.keys()) if rn in mc_scores])
        ml_avg = np.mean([cv_scores[rn] for rn in sorted(cv_scores.keys()) if rn in mc_scores])
        print(f"  {'-'*40}")
        print(f"  {'AVG':>6s} | {mc_avg:8.1f} | {ml_avg:8.1f} | {ml_avg - mc_avg:+8.1f}")
        print(f"\n  ML v2 wins: {sum(1 for d in diffs if d > 0)}/{len(diffs)} rounds")


def main():
    print("=" * 70)
    print("Train Model v2: Enhanced LightGBM for Settlement Prediction")
    print("=" * 70)

    # Step 1: Load data
    print("\n[1/4] Loading training data from cache...")
    all_data = load_all_data()

    # Step 2: Build dataset
    print("\n[2/4] Computing features...")
    X, y, w, groups, meta = build_dataset(all_data)

    # Step 3: Cross-validation
    print("\n[3/4] Leave-one-round-out cross-validation...")
    cv_scores = train_cv(X, y, w, groups, meta)

    # Step 4: Train final model
    print("\n[4/4] Training final model on all data...")
    models = train_final_model(X, y, w)

    # Compare
    compare_with_baselines(cv_scores)

    # Quick sanity: train score
    preds = np.column_stack([m.predict(X) for m in models])
    preds = np.maximum(preds, 0.001)
    preds = preds / preds.sum(axis=-1, keepdims=True)
    train_score = competition_score(y, preds)
    print(f"\nTraining set score (overfit check): {train_score:.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
