"""
Train ML model to predict terrain probability distributions.

Uses per-cell features from initial state + ground truth from analysis endpoint.
Model: LightGBM multi-output (one model per class) with KL-divergence evaluation.
"""

import numpy as np
import requests
import json
import pickle
import time
from collections import defaultdict

import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3ZjdkNzAxZC0xN2I4LTQxYTgtODhlYi05NzYzYjU0MWQ1NTQiLCJlbWFpbCI6ImVybWVzam9hbmRyZWFzQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NjI1NTY0fQ.A-0e_Ga8GZkUbb4NoYJM2Ng-HyLn_B0GeW9Y05KO5ic"
BASE = "https://api.ainm.no"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def fetch_all_data():
    """Fetch initial states and ground truth for all completed rounds."""
    rounds = requests.get(f"{BASE}/astar-island/rounds", headers=HEADERS).json()
    completed = sorted(
        [r for r in rounds if r["status"] == "completed"],
        key=lambda r: r["round_number"],
    )
    print(f"Found {len(completed)} completed rounds")

    all_data = []
    for r in completed:
        round_id = r["id"]
        detail = requests.get(
            f"{BASE}/astar-island/rounds/{round_id}", headers=HEADERS
        ).json()
        W, H = detail["map_width"], detail["map_height"]
        n_seeds = detail["seeds_count"]

        for si in range(n_seeds):
            try:
                analysis = requests.get(
                    f"{BASE}/astar-island/analysis/{round_id}/{si}", headers=HEADERS
                ).json()
                all_data.append({
                    "round_number": r["round_number"],
                    "round_id": round_id,
                    "seed_index": si,
                    "initial_grid": np.array(analysis["initial_grid"]),
                    "ground_truth": np.array(analysis["ground_truth"]),
                    "initial_state": detail["initial_states"][si],
                    "W": W,
                    "H": H,
                })
            except Exception as e:
                print(f"  Round {r['round_number']} seed {si} failed: {e}")

        print(f"  Round {r['round_number']}: {n_seeds} seeds loaded")
        time.sleep(0.2)

    return all_data


def compute_features(initial_grid, settlements_list, W, H):
    """Compute per-cell feature matrix from initial state.

    Returns (H*W, n_features) array.
    """
    grid = np.array(initial_grid) if not isinstance(initial_grid, np.ndarray) else initial_grid
    spos = []
    for s in settlements_list:
        if s.get("alive", True):
            spos.append((s["x"], s["y"]))

    ys, xs = np.mgrid[0:H, 0:W]

    # Distance to nearest settlement
    dist = np.full((H, W), 999.0)
    for sx, sy in spos:
        d = np.abs(xs - sx) + np.abs(ys - sy)
        dist = np.minimum(dist, d.astype(float))

    # Distance to 2nd nearest settlement
    dist2 = np.full((H, W), 999.0)
    for sx, sy in spos:
        d = (np.abs(xs - sx) + np.abs(ys - sy)).astype(float)
        mask = d > dist
        dist2 = np.where(mask, np.minimum(dist2, d), dist2)

    # Coastal mask
    ocean = grid == 10
    land = ~ocean
    coastal = np.zeros((H, W), dtype=float)
    if H > 1:
        coastal[1:, :] += land[1:, :] & ocean[:-1, :]
        coastal[:-1, :] += land[:-1, :] & ocean[1:, :]
    if W > 1:
        coastal[:, 1:] += land[:, 1:] & ocean[:, :-1]
        coastal[:, :-1] += land[:, :-1] & ocean[:, 1:]

    # Settlement counts at various radii
    n_r1 = np.zeros((H, W), dtype=float)
    n_r2 = np.zeros((H, W), dtype=float)
    n_r3 = np.zeros((H, W), dtype=float)
    n_r5 = np.zeros((H, W), dtype=float)
    n_r8 = np.zeros((H, W), dtype=float)
    for sx, sy in spos:
        d = np.abs(xs - sx) + np.abs(ys - sy)
        n_r1 += (d <= 1)
        n_r2 += (d <= 2)
        n_r3 += (d <= 3)
        n_r5 += (d <= 5)
        n_r8 += (d <= 8)

    # Adjacent terrain counts (4-connected)
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

    adj_forest = count_adjacent(4)
    adj_ocean = count_adjacent(10)
    adj_settlement = count_adjacent(1) + count_adjacent(2)
    adj_plains = count_adjacent(11)

    # Terrain one-hot
    is_ocean = (grid == 10).astype(float)
    is_plains = (grid == 11).astype(float)
    is_forest = (grid == 4).astype(float)
    is_mountain = (grid == 5).astype(float)
    is_settlement = (grid == 1).astype(float)
    is_port = (grid == 2).astype(float)

    # Global features (same for all cells in a seed)
    total_settlements = float(len(spos))
    total_land = float(np.sum(land))
    settlement_density = total_settlements / max(total_land, 1)

    # Distance to map edge
    dist_edge = np.minimum(
        np.minimum(xs, W - 1 - xs),
        np.minimum(ys, H - 1 - ys)
    ).astype(float)

    # Land connectivity: how much land is within radius 5
    land_r5 = np.zeros((H, W), dtype=float)
    land_float = land.astype(float)
    # Approximate with box filter
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            if abs(dy) + abs(dx) <= 5:
                shifted = np.roll(np.roll(land_float, dy, axis=0), dx, axis=1)
                land_r5 += shifted

    # Stack features
    features = np.stack([
        is_ocean, is_plains, is_forest, is_mountain, is_settlement, is_port,  # 0-5
        dist, dist2,  # 6-7
        coastal,  # 8
        n_r1, n_r2, n_r3, n_r5, n_r8,  # 9-13
        adj_forest, adj_ocean, adj_settlement, adj_plains,  # 14-17
        dist_edge,  # 18
        land_r5,  # 19
        np.full((H, W), total_settlements),  # 20
        np.full((H, W), settlement_density),  # 21
    ], axis=-1)  # (H, W, n_features)

    return features.reshape(H * W, -1)


FEATURE_NAMES = [
    "is_ocean", "is_plains", "is_forest", "is_mountain", "is_settlement", "is_port",
    "dist_nearest", "dist_2nd_nearest",
    "coastal",
    "n_settle_r1", "n_settle_r2", "n_settle_r3", "n_settle_r5", "n_settle_r8",
    "adj_forest", "adj_ocean", "adj_settlement", "adj_plains",
    "dist_edge",
    "land_r5",
    "total_settlements", "settlement_density",
]


def build_dataset(all_data):
    """Build training dataset from all round data."""
    X_all = []
    y_all = []
    groups = []  # round number for leave-one-round-out CV

    for d in all_data:
        grid = d["initial_grid"]
        gt = d["ground_truth"]
        W, H = d["W"], d["H"]

        # Reconstruct settlements list
        settlements = []
        for y in range(H):
            for x in range(W):
                v = grid[y, x]
                if v in (1, 2):
                    settlements.append({"x": x, "y": y, "has_port": v == 2, "alive": True})

        features = compute_features(grid, settlements, W, H)
        targets = gt.reshape(H * W, -1)

        X_all.append(features)
        y_all.append(targets)
        groups.extend([d["round_number"]] * (H * W))

    X = np.vstack(X_all)
    y = np.vstack(y_all)
    groups = np.array(groups)

    print(f"Dataset: {X.shape[0]} cells, {X.shape[1]} features, {y.shape[1]} classes")
    return X, y, groups


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


def train_model(X, y, groups):
    """Train LightGBM models (one per class) with leave-one-round-out CV."""
    n_classes = y.shape[1]
    unique_rounds = sorted(set(groups))

    # Cross-validation scores
    cv_scores = []
    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(X, y, groups):
        test_round = groups[test_idx[0]]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train one model per class
        preds = np.zeros_like(y_test)
        for c in range(n_classes):
            model = lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=50,
                reg_alpha=0.1,
                reg_lambda=0.1,
                verbose=-1,
            )
            model.fit(X_train, y_train[:, c])
            preds[:, c] = model.predict(X_test)

        # Clip and normalize
        preds = np.maximum(preds, 0.001)
        preds = preds / preds.sum(axis=-1, keepdims=True)

        # Score per seed (1600 cells each)
        n_cells = 1600
        n_test_seeds = len(test_idx) // n_cells
        seed_scores = []
        for s in range(n_test_seeds):
            start = s * n_cells
            end = (s + 1) * n_cells
            sc = competition_score(y_test[start:end], preds[start:end])
            seed_scores.append(sc)

        avg = np.mean(seed_scores)
        cv_scores.append(avg)
        print(f"  Round {test_round} (held out): avg score = {avg:.1f} ({', '.join(f'{s:.1f}' for s in seed_scores)})")

    print(f"\nCV average: {np.mean(cv_scores):.1f}")
    return cv_scores


def train_final_model(X, y):
    """Train final model on all data and save."""
    n_classes = y.shape[1]
    models = []

    for c in range(n_classes):
        model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=0.1,
            verbose=-1,
        )
        model.fit(X, y[:, c])
        models.append(model)
        print(f"  Trained class {c} ({CLASS_NAMES[c]})")

    # Feature importance
    print("\nTop features per class:")
    for c, model in enumerate(models):
        imp = model.feature_importances_
        top = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])[:5]
        desc = ", ".join(f"{n}={v:.0f}" for n, v in top)
        print(f"  {CLASS_NAMES[c]}: {desc}")

    # Save models
    with open("ml_models.pkl", "wb") as f:
        pickle.dump(models, f)
    print("\nModels saved to ml_models.pkl")

    return models


def main():
    print("=== Fetching training data ===")
    all_data = fetch_all_data()

    print("\n=== Building dataset ===")
    X, y, groups = build_dataset(all_data)

    print("\n=== Cross-validation (leave-one-round-out) ===")
    cv_scores = train_model(X, y, groups)

    print("\n=== Training final model on all data ===")
    models = train_final_model(X, y)

    # Quick sanity check: predict on training data
    preds = np.column_stack([m.predict(X) for m in models])
    preds = np.maximum(preds, 0.001)
    preds = preds / preds.sum(axis=-1, keepdims=True)
    train_score = competition_score(y, preds)
    print(f"\nTraining set score: {train_score:.1f}")


if __name__ == "__main__":
    main()
