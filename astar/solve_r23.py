#!/usr/bin/env python3
"""
solve_r23.py — R23 observation-heavy strategy
==============================================
Maximizes coverage via many unique viewports with minimal repeats.
Uses ML v2 model as primary predictor, MC simulation as fallback only.

Strategy:
  1. Check for active round R23
  2. Scout: 2 queries/seed (10 total) → detect settle_rate
  3. Coverage: 40 queries spread as 8 unique viewports × 1 each per seed
     (smart placement avoiding ocean, focusing on settlement-dense areas)
  4. Prediction pipeline (per cell):
     - 2+ observations: empirical distribution (sharp!)
     - 1 observation:   70% ML_v2 + 30% empirical
     - 0 observations:  ML_v2 prediction (uses settle_rate as feature)
  5. Fallback (no ML v2): MC-only with interpolated params
  6. Submit all 5 seeds
  7. Re-submit loop every 10 min with refined params

Usage:
  python solve_r23.py              # run once
  python solve_r23.py --loop       # run + re-submit every 10 min
  python solve_r23.py --dry-run    # check round status only
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [R23] %(message)s",
    handlers=[
        logging.FileHandler("solve_r23.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("solve_r23")

# Import from existing codebase
sys.path.insert(0, os.path.dirname(__file__) or ".")
from solver import AstarSolver, TERRAIN_TO_CLASS, CLASS_NAMES, params_from_rate
from simulator import AstarSimulator, SimParams
from sim_params import NORMAL_PARAMS, DIEOFF_PARAMS

TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI3ZjdkNzAxZC0xN2I4LTQxYTgtODhlYi05NzYzYjU0MWQ1NTQiLCJlbWFpbCI6ImVybWVzam9hbmRyZWFzQGdtYWlsLmNvbSIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NjI1NTY0fQ.A-0e_Ga8GZkUbb4NoYJM2Ng-HyLn_B0GeW9Y05KO5ic"
BASE = "https://api.ainm.no"


# ---------------------------------------------------------------------------
# ML v2 model loading
# ---------------------------------------------------------------------------
def load_ml_v2():
    """Load ML v2 models (LightGBM, 6 classes). Returns list of models or None."""
    path = os.path.join(os.path.dirname(__file__) or ".", "ml_models_v2.pkl")
    try:
        with open(path, "rb") as f:
            models = pickle.load(f)
        log.info("Loaded ML v2: %d models from %s", len(models), path)
        return models
    except Exception as e:
        log.warning("ML v2 not available (%s) — will use fallback", e)
        return None


def load_ml_v1():
    """Load ML v1 models as secondary fallback."""
    path = os.path.join(os.path.dirname(__file__) or ".", "ml_models.pkl")
    try:
        with open(path, "rb") as f:
            models = pickle.load(f)
        log.info("Loaded ML v1: %d models from %s", len(models), path)
        return models
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ML v2 prediction
# ---------------------------------------------------------------------------
def predict_ml_v2(models, grid, settlements_list, W, H, settle_rate=0.0):
    """Run ML v2 prediction. Returns (H, W, 6) probability array."""
    from train_model import compute_features
    features = compute_features(grid, settlements_list, W, H)
    # features is (H*W, n_features)

    # Add settle_rate as an extra feature if the model was trained with it
    # For safety, check feature count matches model expectation
    n_model_features = models[0].n_features_
    if features.shape[1] < n_model_features:
        # Pad with settle_rate and potentially other derived features
        extra = np.full((features.shape[0], n_model_features - features.shape[1]), settle_rate)
        features = np.hstack([features, extra])
    elif features.shape[1] > n_model_features:
        features = features[:, :n_model_features]

    preds = np.column_stack([m.predict(features) for m in models])
    preds = np.maximum(preds, 0.001)
    preds = preds / preds.sum(axis=-1, keepdims=True)
    return preds.reshape(H, W, 6)


# ---------------------------------------------------------------------------
# Coverage-maximizing viewport planning
# ---------------------------------------------------------------------------
VP_MAX = 15


def plan_coverage_viewports(analysis, n_queries, W, H, existing_coverage=None):
    """Plan viewports to maximize unique cell coverage.

    Unlike the base solver which weights by dynamism x coverage-count,
    this version strongly penalizes ANY repeat coverage to push for
    maximum unique cell reach.

    Args:
        analysis: seed analysis dict from AstarSolver.analyse_seed
        n_queries: number of viewports to place
        W, H: map dimensions
        existing_coverage: (H, W) int array of prior observation counts (optional)

    Returns:
        List of (x, y, w, h) tuples
    """
    dyn = analysis["dynamic"]
    grid = analysis["grid"]
    ocean = (grid == 10)

    # Start from existing coverage or fresh
    coverage = existing_coverage.copy() if existing_coverage is not None else np.zeros((H, W), dtype=int)

    # Candidate viewport positions — finer grid than default solver for better packing
    max_sx = max(0, W - 5)
    max_sy = max(0, H - 5)
    # Use step=1 for exhaustive search on small maps, step=2 otherwise
    step = 1 if W * H <= 2500 else 2
    pos_x = sorted(set(list(range(0, max_sx + 1, step)) + [max_sx]))
    pos_y = sorted(set(list(range(0, max_sy + 1, step)) + [max_sy]))

    viewports = []
    for _ in range(n_queries):
        best, best_score = None, -1.0
        for vy in pos_y:
            for vx in pos_x:
                vw = min(VP_MAX, W - vx)
                vh = min(VP_MAX, H - vy)
                if vw < 5 or vh < 5:
                    continue

                region_dyn = dyn[vy:vy + vh, vx:vx + vw]
                region_cov = coverage[vy:vy + vh, vx:vx + vw]
                region_ocean = ocean[vy:vy + vh, vx:vx + vw]

                # Score: strongly reward NEW cells, penalize ocean-heavy areas
                new_cells = (region_cov == 0) & ~region_ocean
                n_new = int(new_cells.sum())
                n_new_dynamic = float((region_dyn * new_cells).sum())

                # Count non-ocean land cells that are new
                n_ocean = int(region_ocean.sum())
                ocean_fraction = n_ocean / max(region_dyn.size, 1)

                # Primary: maximize new dynamic cell coverage
                # Secondary: prefer less ocean coverage
                # Tertiary: penalize areas already covered
                already_covered = int((region_cov > 0).sum())

                score = (
                    n_new_dynamic * 10.0      # dynamic new cells (highest priority)
                    + n_new * 2.0              # any new cell
                    - already_covered * 5.0    # penalize re-covering
                    - ocean_fraction * 3.0     # penalize ocean-heavy viewports
                )

                if score > best_score:
                    best_score = score
                    best = (vx, vy, vw, vh)

        if best is None or best_score <= 0:
            # If no good viewport left, fall back to least-covered area
            if best is None:
                break
        vx, vy, vw, vh = best
        coverage[vy:vy + vh, vx:vx + vw] += 1
        viewports.append(best)

    return viewports


# ---------------------------------------------------------------------------
# Observation tracking
# ---------------------------------------------------------------------------
class ObservationTracker:
    """Track per-cell observations across all queries for a round."""

    def __init__(self, n_seeds, W, H):
        self.n_seeds = n_seeds
        self.W = W
        self.H = H
        # cell_counts[seed][(y, x)] = Counter of terrain classes
        self.cell_counts = [defaultdict(Counter) for _ in range(n_seeds)]
        # cell_obs_count[seed][(y, x)] = int
        self.cell_obs_count = [defaultdict(int) for _ in range(n_seeds)]
        # coverage[seed] = (H, W) int array of observation count
        self.coverage = [np.zeros((H, W), dtype=int) for _ in range(n_seeds)]
        # Store raw observations for solver compatibility
        self.raw_obs = []

    def add_observation(self, seed_idx, viewport, grid_2d):
        """Record a viewport observation."""
        vx, vy = viewport["x"], viewport["y"]
        for dy, row in enumerate(grid_2d):
            for dx, val in enumerate(row):
                y = vy + dy
                x = vx + dx
                if 0 <= y < self.H and 0 <= x < self.W:
                    cls = TERRAIN_TO_CLASS.get(val, 0)
                    self.cell_counts[seed_idx][(y, x)][cls] += 1
                    self.cell_obs_count[seed_idx][(y, x)] += 1
                    self.coverage[seed_idx][y, x] += 1

        # Store raw observation
        self.raw_obs.append({
            "seed": seed_idx,
            "viewport": viewport,
            "grid": grid_2d,
        })

    def get_obs_count(self, seed_idx, y, x):
        """How many times was this cell observed?"""
        return self.cell_obs_count[seed_idx].get((y, x), 0)

    def get_empirical_dist(self, seed_idx, y, x):
        """Get empirical probability distribution for a cell. Returns (6,) array or None."""
        key = (y, x)
        if key not in self.cell_counts[seed_idx]:
            return None
        counts = self.cell_counts[seed_idx][key]
        total = sum(counts.values())
        if total == 0:
            return None
        probs = np.zeros(6)
        for cls, cnt in counts.items():
            probs[cls] = cnt / total
        return probs

    def total_observed_cells(self, seed_idx):
        """Count of unique cells observed at least once."""
        return int((self.coverage[seed_idx] > 0).sum())

    def total_multi_observed_cells(self, seed_idx):
        """Count of unique cells observed 2+ times."""
        return int((self.coverage[seed_idx] >= 2).sum())

    def compute_settle_rate(self, seed_idx=None):
        """Compute settlement rate from all observations."""
        total_cells = 0
        total_settle = 0
        seeds = range(self.n_seeds) if seed_idx is None else [seed_idx]
        for si in seeds:
            for (y, x), counter in self.cell_counts[si].items():
                total = sum(counter.values())
                total_cells += total
                total_settle += counter.get(1, 0) + counter.get(2, 0)
        return total_settle / max(total_cells, 1)

    def get_solver_format_obs(self):
        """Return observations in the format solver.py expects."""
        return list(self.raw_obs)


# ---------------------------------------------------------------------------
# R23 prediction pipeline
# ---------------------------------------------------------------------------
MIN_PROB = 0.001


def build_r23_prediction(
    seed_idx: int,
    analysis: dict,
    tracker: ObservationTracker,
    ml_v2_models,
    settle_rate: float,
    W: int, H: int,
    solver: AstarSolver,
    mc_pred: Optional[np.ndarray] = None,
):
    """Build prediction for one seed using R23 observation-heavy strategy.

    Pipeline per cell:
      - 2+ observations: empirical distribution (sharp, high confidence)
      - 1 observation: 70% ML_v2 + 30% empirical
      - 0 observations: ML_v2 prediction

    Fallback (no ML v2): MC-only with interpolated params.
    """
    grid = analysis["grid"]
    dist = analysis["dist"]
    coastal = analysis["coastal"]

    use_ml = ml_v2_models is not None
    ml_pred = None

    # ---- Try ML prediction ----
    if use_ml:
        settlements_list = analysis.get("settlements", [])
        try:
            ml_pred = predict_ml_v2(ml_v2_models, grid, settlements_list, W, H, settle_rate)
            log.info("  Seed %d: ML prediction computed successfully", seed_idx)
        except Exception as e:
            log.warning("  Seed %d: ML prediction failed: %s — falling back to MC", seed_idx, e)
            use_ml = False
            ml_pred = None

    # ---- PATH A: ML available ----
    if use_ml and ml_pred is not None:
        log.info("  Seed %d: using ML v2 pipeline", seed_idx)

        pred = np.zeros((H, W, 6))
        n_emp2 = 0  # cells with 2+ obs
        n_emp1 = 0  # cells with 1 obs
        n_ml = 0    # cells with 0 obs

        for y in range(H):
            for x in range(W):
                tv = int(grid[y, x])

                # Static terrains: always deterministic
                if tv == 10:  # Ocean
                    pred[y, x] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    continue
                if tv == 5:   # Mountain
                    pred[y, x] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    continue

                obs_count = tracker.get_obs_count(seed_idx, y, x)
                emp_dist = tracker.get_empirical_dist(seed_idx, y, x)

                if obs_count >= 2 and emp_dist is not None:
                    # 2+ observations: trust empirical distribution (sharp!)
                    # Small ML smoothing to avoid 0-probability traps
                    smoothed = emp_dist * 0.95 + ml_pred[y, x] * 0.05
                    pred[y, x] = smoothed
                    n_emp2 += 1
                elif obs_count == 1 and emp_dist is not None:
                    # 1 observation: blend 70% ML + 30% empirical
                    # Single obs is noisy but informative directionally
                    pred[y, x] = 0.70 * ml_pred[y, x] + 0.30 * emp_dist
                    n_emp1 += 1
                else:
                    # 0 observations: pure ML
                    pred[y, x] = ml_pred[y, x]
                    n_ml += 1

        log.info("  Seed %d cell breakdown: emp2+=%d, emp1=%d, ml_only=%d",
                 seed_idx, n_emp2, n_emp1, n_ml)

    # ---- PATH B: Fallback to MC + priors (no ML) ----
    else:
        log.info("  Seed %d: FALLBACK — MC + priors (no ML)", seed_idx)

        n_nearby = analysis.get("n_nearby", np.zeros_like(grid))
        n_local = analysis.get("n_local", np.zeros_like(grid))
        adj_forest = analysis.get("adj_forest", np.zeros_like(grid))

        # Build prior-based prediction using solver's _prior method
        pred = np.zeros((H, W, 6))
        for y in range(H):
            for x in range(W):
                tv = int(grid[y, x])
                d = float(dist[y, x])
                c = bool(coastal[y, x])
                alphas = solver._prior(tv, d, c, int(n_nearby[y, x]),
                                       int(adj_forest[y, x]), int(n_local[y, x]))
                pred[y, x] = alphas / alphas.sum()

        # Blend with MC if available
        if mc_pred is not None:
            is_dynamic = (dist <= solver.DYNAMIC_RANGE) & (grid != 10) & (grid != 5)
            mc_w = 0.50
            prior_w = 0.50
            blend = mc_w * mc_pred + prior_w * pred
            mask_3d = is_dynamic[:, :, np.newaxis]
            pred = np.where(mask_3d, blend, pred)

        # Apply empirical overlay (conservative 30% weight for fallback)
        for y in range(H):
            for x in range(W):
                obs_count = tracker.get_obs_count(seed_idx, y, x)
                if obs_count >= 2:
                    emp_dist = tracker.get_empirical_dist(seed_idx, y, x)
                    if emp_dist is not None:
                        pred[y, x] = 0.70 * pred[y, x] + 0.30 * emp_dist

    # ---- Common post-processing ----
    # Wavefront smoothing (gentle)
    settle_prob = pred[:, :, 1].copy()
    boost = np.zeros_like(settle_prob)
    if H > 1:
        boost[1:] += settle_prob[:-1]
        boost[:-1] += settle_prob[1:]
    if W > 1:
        boost[:, 1:] += settle_prob[:, :-1]
        boost[:, :-1] += settle_prob[:, 1:]
    wf_factor = 0.02  # gentle wavefront
    not_static = (grid != 10) & (grid != 5)
    pred[:, :, 1] += boost * not_static * wf_factor
    pred[:, :, 0] -= boost * not_static * wf_factor * 0.7
    pred[:, :, 4] -= boost * not_static * wf_factor * 0.3

    # Floor and renormalize
    pred = np.maximum(pred, MIN_PROB)
    pred /= pred.sum(axis=-1, keepdims=True)

    return pred


# ---------------------------------------------------------------------------
# Main R23 solver
# ---------------------------------------------------------------------------
class R23Solver:
    """R23-specific solver with observation-heavy strategy."""

    SCOUT_PER_SEED = 2       # 10 total scout queries
    COVERAGE_BUDGET = 40     # 40 queries for coverage viewports
    SIMULATE_DELAY = 0.25    # stay under 5 req/s
    SUBMIT_DELAY = 0.55      # stay under 2 req/s

    def __init__(self):
        self.solver = AstarSolver(TOKEN, use_mc=True)
        self.ml_v2 = load_ml_v2()
        self.ml_v1 = load_ml_v1() if self.ml_v2 is None else None
        self.session = self.solver.session

    def _get(self, path):
        return self.solver._get(path)

    def _post(self, path, body):
        return self.solver._post(path, body)

    def find_active_round(self, expected_number=23):
        """Find active round. Optionally verify it's R23."""
        rounds = self._get("/astar-island/rounds")
        active = [r for r in rounds if r["status"] == "active"]
        if not active:
            log.error("No active round found!")
            for r in sorted(rounds, key=lambda x: x.get("round_number", 0))[-5:]:
                log.info("  R%s: %s", r.get("round_number"), r["status"])
            return None

        r = active[0]
        rn = r.get("round_number", "?")
        log.info("Active round: R%s (id=%s)", rn, r["id"][:8])
        if expected_number and rn != expected_number:
            log.warning("Expected R%d but found R%s — proceeding anyway", expected_number, rn)
        return r

    def get_budget_remaining(self):
        """Get remaining query budget."""
        try:
            info = self._get("/astar-island/budget")
            used = info.get("queries_used", 0)
            total = info.get("queries_max", 50)
            return total - used, total
        except Exception as e:
            log.warning("Budget check failed: %s", e)
            return 50, 50  # assume full budget

    def run_query(self, round_id, seed_idx, vx, vy, vw, vh):
        """Execute a single simulate query."""
        result = self._post("/astar-island/simulate", {
            "round_id": round_id,
            "seed_index": seed_idx,
            "viewport_x": vx,
            "viewport_y": vy,
            "viewport_w": vw,
            "viewport_h": vh,
        })
        time.sleep(self.SIMULATE_DELAY)
        return result

    def submit_prediction(self, round_id, seed_idx, prediction):
        """Submit prediction for one seed."""
        result = self._post("/astar-island/submit", {
            "round_id": round_id,
            "seed_index": seed_idx,
            "prediction": prediction,
        })
        time.sleep(self.SUBMIT_DELAY)
        return result

    # ------------------------------------------------------------------
    # Phase 1: Scout queries
    # ------------------------------------------------------------------
    def run_scout_phase(self, round_id, analyses, tracker, n_seeds, W, H):
        """2 queries per seed to detect settle rate and round type."""
        log.info("=== PHASE 1: Scout (2/seed = %d queries) ===", self.SCOUT_PER_SEED * n_seeds)

        scout_plan = self.solver.plan_scout_queries(analyses, n_seeds, W, H)
        queries_used = 0

        for i, q in enumerate(scout_plan):
            try:
                result = self.run_query(
                    round_id, q["seed"], q["x"], q["y"], q["w"], q["h"]
                )
                tracker.add_observation(q["seed"], result["viewport"], result["grid"])
                queries_used += 1
                log.info("  Scout %d/%d  seed=%d vp=(%d,%d %dx%d)",
                         i + 1, len(scout_plan), q["seed"],
                         q["x"], q["y"], q["w"], q["h"])
            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code == 429:
                    log.warning("Rate limit hit during scout — stopping")
                    break
                log.error("Scout HTTP %d: %s", code, e)
                break
            except Exception as e:
                log.error("Scout query failed: %s", e)
                break

        # Detect settle rate
        settle_rate = tracker.compute_settle_rate()
        log.info("Scout settle_rate = %.4f", settle_rate)

        # Detect round type from settle rate
        if settle_rate < 0.03:
            round_type = "dieoff"
        elif settle_rate > 0.08:
            round_type = "normal"
        else:
            round_type = "normal"  # conservative default
        log.info("Round type: %s", round_type)

        self.solver.round_type = round_type
        self.solver._observed_settle_rate = settle_rate

        return settle_rate, round_type, queries_used

    # ------------------------------------------------------------------
    # Phase 2: Coverage queries
    # ------------------------------------------------------------------
    def run_coverage_phase(self, round_id, analyses, tracker, n_seeds, W, H, budget):
        """Maximum coverage: spread unique viewports across the map."""
        # Distribute budget across seeds proportionally to dynamism
        seed_dynamism = []
        for si in range(n_seeds):
            dyn_sum = float(analyses[si]["dynamic"].sum())
            # Boost seeds with observed settlements
            obs_settle = 0
            for (y, x), counter in tracker.cell_counts[si].items():
                obs_settle += counter.get(1, 0) + counter.get(2, 0)
            dyn_sum += obs_settle * 5
            seed_dynamism.append(dyn_sum)

        total_dyn = sum(seed_dynamism) or 1.0
        alloc = [max(2, int(budget * d / total_dyn)) for d in seed_dynamism]

        # Normalize to budget
        while sum(alloc) > budget:
            biggest = max(range(n_seeds), key=lambda i: alloc[i])
            alloc[biggest] -= 1
        while sum(alloc) < budget:
            smallest = min(range(n_seeds), key=lambda i: alloc[i])
            alloc[smallest] += 1

        log.info("=== PHASE 2: Coverage (%d queries) ===", budget)
        log.info("  Allocation: %s (dynamism: %s)",
                 alloc, [f"{d:.0f}" for d in seed_dynamism])

        queries_used = 0
        for si in range(n_seeds):
            if alloc[si] <= 0:
                continue

            # Plan viewports avoiding already-covered areas
            vps = plan_coverage_viewports(
                analyses[si], alloc[si], W, H,
                existing_coverage=tracker.coverage[si]
            )

            for j, (vx, vy, vw, vh) in enumerate(vps):
                try:
                    result = self.run_query(round_id, si, vx, vy, vw, vh)
                    tracker.add_observation(si, result["viewport"], result["grid"])
                    queries_used += 1
                    log.info("  Coverage seed=%d %d/%d vp=(%d,%d %dx%d)",
                             si, j + 1, len(vps), vx, vy, vw, vh)
                except requests.exceptions.HTTPError as e:
                    code = e.response.status_code if e.response is not None else 0
                    if code == 429:
                        log.warning("Rate/budget limit hit — stopping coverage")
                        return queries_used
                    log.error("Coverage HTTP %d: %s", code, e)
                    return queries_used
                except Exception as e:
                    log.error("Coverage query failed: %s", e)
                    return queries_used

        # Log coverage stats
        for si in range(n_seeds):
            total_cells = W * H
            observed = tracker.total_observed_cells(si)
            multi = tracker.total_multi_observed_cells(si)
            log.info("  Seed %d coverage: %d/%d cells (%.0f%%), %d with 2+ obs",
                     si, observed, total_cells, 100 * observed / total_cells, multi)

        return queries_used

    # ------------------------------------------------------------------
    # Phase 3: Build and submit predictions
    # ------------------------------------------------------------------
    def build_and_submit(self, round_id, analyses, tracker, states,
                         settle_rate, n_seeds, W, H, label="initial"):
        """Build predictions for all seeds and submit."""
        log.info("=== SUBMIT (%s) ===", label)

        sim_params = params_from_rate(settle_rate)
        log.info("Interpolated params: sp=%.3f db=%.4f fbr=%.3f",
                 sim_params.spawn_prob, sim_params.death_base_rate, sim_params.food_base_regen)

        # Choose ML models: prefer v2, then v1, then None (MC fallback)
        ml_models = self.ml_v2 if self.ml_v2 is not None else self.ml_v1

        results = []
        for si in range(n_seeds):
            # Run MC prediction only if no ML available (fallback path)
            mc_pred = None
            if ml_models is None:
                mc_pred = self.solver._run_mc_prediction(
                    analyses[si], sim_params, n_runs=500, timeout=120.0
                )

            pred = build_r23_prediction(
                seed_idx=si,
                analysis=analyses[si],
                tracker=tracker,
                ml_v2_models=ml_models,
                settle_rate=settle_rate,
                W=W, H=H,
                solver=self.solver,
                mc_pred=mc_pred,
            )

            try:
                resp = self.submit_prediction(round_id, si, pred.tolist())
                status = resp.get("status", resp.get("score", "ok"))
                results.append({"seed": si, "status": status})
                log.info("  Submitted seed %d: %s", si, status)
            except Exception as e:
                results.append({"seed": si, "error": str(e)})
                log.error("  Submit seed %d failed: %s", si, e)

        return results

    # ------------------------------------------------------------------
    # Phase 4: Re-submit loop
    # ------------------------------------------------------------------
    def resubmit_loop(self, round_id, analyses, tracker, states,
                      settle_rate, n_seeds, W, H, interval_sec=600, max_iter=30):
        """Re-submit with refined params every interval_sec seconds."""
        log.info("=== RE-SUBMIT LOOP (every %ds) ===", interval_sec)

        for iteration in range(1, max_iter + 1):
            log.info("Waiting %d seconds before re-submit #%d...", interval_sec, iteration)
            time.sleep(interval_sec)

            # Check if round is still active
            try:
                rounds = self._get("/astar-island/rounds")
                active = [r for r in rounds if r["status"] == "active" and r["id"] == round_id]
                if not active:
                    log.info("Round no longer active — stopping re-submit loop")
                    break
            except Exception as e:
                log.warning("Round check failed: %s — continuing anyway", e)

            # Re-submit with same observations (predictions don't change without new obs,
            # but re-submitting ensures we have latest prediction on the board)
            log.info("--- Re-submit iteration %d ---", iteration)
            self.build_and_submit(
                round_id, analyses, tracker, states,
                settle_rate, n_seeds, W, H,
                label=f"resubmit-{iteration}"
            )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def solve(self, dry_run=False, loop=False, expected_round=None):
        """Run the full R23 solve pipeline."""
        log.info("=" * 60)
        log.info("R23 Solver — observation-heavy + ML v2 strategy")
        log.info("=" * 60)

        # 1. Find active round
        rinfo = self.find_active_round(expected_number=expected_round)
        if rinfo is None:
            return {"error": "No active round"}

        round_id = rinfo["id"]
        round_number = rinfo.get("round_number", "?")

        if dry_run:
            remaining, total = self.get_budget_remaining()
            log.info("DRY RUN: R%s active, budget=%d/%d, ML_v2=%s",
                     round_number, remaining, total,
                     "available" if self.ml_v2 else "NOT available")
            return {"status": "dry_run", "round": round_number, "budget": remaining}

        # 2. Fetch round details
        detail = self.solver.get_round_detail(round_id)
        W = detail["map_width"]
        H = detail["map_height"]
        n_seeds = detail["seeds_count"]
        states = detail["initial_states"]
        log.info("Map %dx%d, %d seeds", W, H, n_seeds)

        # 3. Analyze all seeds
        analyses = [self.solver.analyse_seed(s, W, H) for s in states]

        # 4. Initialize observation tracker
        tracker = ObservationTracker(n_seeds, W, H)

        # Load cached observations if any
        cached_obs = self.solver._load_observations(round_id)
        if cached_obs:
            log.info("Loading %d cached observations", len(cached_obs))
            for obs in cached_obs:
                tracker.add_observation(obs["seed"], obs["viewport"], obs["grid"])

        # 5. Check budget
        remaining, total = self.get_budget_remaining()
        log.info("Budget: %d/%d remaining", remaining, total)

        if remaining <= 0:
            log.info("No budget — submitting with cached data only")
            settle_rate = tracker.compute_settle_rate()
            if settle_rate == 0:
                settle_rate = 0.10  # default guess
            self.build_and_submit(
                round_id, analyses, tracker, states,
                settle_rate, n_seeds, W, H, label="cached-only"
            )
            if loop:
                self.resubmit_loop(
                    round_id, analyses, tracker, states,
                    settle_rate, n_seeds, W, H
                )
            return {"status": "submitted_cached", "round_id": round_id}

        # 6. Phase 1: Scout queries (10 queries)
        settle_rate, round_type, scout_used = self.run_scout_phase(
            round_id, analyses, tracker, n_seeds, W, H
        )
        remaining -= scout_used

        # 7. Safety submit — get something on the board ASAP
        log.info("--- Safety submit after scout ---")
        self.build_and_submit(
            round_id, analyses, tracker, states,
            settle_rate, n_seeds, W, H, label="safety"
        )

        # 8. Phase 2: Coverage queries (use remaining budget)
        if remaining > 0:
            coverage_budget = min(remaining, self.COVERAGE_BUDGET)
            coverage_used = self.run_coverage_phase(
                round_id, analyses, tracker, n_seeds, W, H, coverage_budget
            )
            remaining -= coverage_used

            # Update settle rate with new observations
            settle_rate = tracker.compute_settle_rate()
            log.info("Updated settle_rate after coverage: %.4f", settle_rate)
            self.solver._observed_settle_rate = settle_rate

        # 9. Save observations for future re-use
        self.solver._save_observations(round_id, tracker.get_solver_format_obs())
        self._save_tracker_cache(round_id, tracker)

        # 10. Final submit with full observations
        results = self.build_and_submit(
            round_id, analyses, tracker, states,
            settle_rate, n_seeds, W, H, label="final"
        )

        # 11. Re-submit loop if requested
        if loop:
            self.resubmit_loop(
                round_id, analyses, tracker, states,
                settle_rate, n_seeds, W, H,
                interval_sec=600, max_iter=30  # 5 hours max
            )

        total_obs = sum(tracker.total_observed_cells(si) for si in range(n_seeds))
        return {
            "status": "completed",
            "round_id": round_id,
            "round_number": round_number,
            "settle_rate": settle_rate,
            "round_type": round_type,
            "ml_v2_used": self.ml_v2 is not None,
            "total_observed_cells": total_obs,
            "results": results,
        }

    def _save_tracker_cache(self, round_id, tracker):
        """Save tracker state for potential re-use."""
        cache_dir = os.path.join(os.path.dirname(__file__) or ".", "obs_cache")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{round_id}_r23_tracker.json")

        data = {
            "n_seeds": tracker.n_seeds,
            "W": tracker.W,
            "H": tracker.H,
            "cell_data": {},
        }
        for si in range(tracker.n_seeds):
            seed_data = {}
            for (y, x), counter in tracker.cell_counts[si].items():
                seed_data[f"{y},{x}"] = dict(counter)
            data["cell_data"][str(si)] = seed_data

        with open(path, "w") as f:
            json.dump(data, f)
        log.info("Saved tracker cache to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="R23 observation-heavy solver")
    parser.add_argument("--loop", action="store_true",
                        help="Re-submit every 10 min after initial solve")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check round status only, don't query or submit")
    parser.add_argument("--round", type=int, default=None,
                        help="Expected round number (default: any active round)")
    parser.add_argument("--interval", type=int, default=600,
                        help="Re-submit interval in seconds (default: 600)")
    args = parser.parse_args()

    solver = R23Solver()
    result = solver.solve(
        dry_run=args.dry_run,
        loop=args.loop,
        expected_round=args.round,
    )

    log.info("Result: %s", json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
