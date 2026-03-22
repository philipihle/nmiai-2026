"""
Astar Island Solver — Unified (normal + die-off)
=================================================
Observes a Norse civilisation simulator through viewports and predicts
the probability distribution of terrain types across the full map.

Strategy:
1. Analyse initial states to classify cells as static or dynamic
2. Run MC simulation as safety net and submit immediately
3. Scout queries to detect round type (normal vs die-off)
4. MC-informed predictions blended with priors and ML
5. Deep queries for refinement, final MC + blend → submit
"""

import logging
import os
import pickle
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from simulator_exp import AstarSimulator, SimParams, _grid_to_counts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid value → prediction class index
# ---------------------------------------------------------------------------
TERRAIN_TO_CLASS = {
    10: 0,  # Ocean   → Empty
    11: 0,  # Plains  → Empty
    0: 0,   # Empty   → Empty
    1: 1,   # Settlement
    2: 2,   # Port
    3: 3,   # Ruin
    4: 4,   # Forest
    5: 5,   # Mountain
}

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# --- Per-round parameter interpolation ---
# Calibration points from tune_per_round.py, sorted by settle_rate
_CALIBRATION_POINTS = [
    {"rate": 0.021, "round": 3, "params": {
        "spawn_prob": 0.15, "spawn_pop_threshold": 0.3, "death_base_rate": 0.150,
        "death_food_factor": 0.15, "pop_growth_rate": 0.08, "food_base_regen": 0.15,
        "food_competition": 0.05, "food_pop_drain": 0.08, "food_forest_bonus": 0.02,
        "port_survival_bonus": 0.03, "port_wealth_threshold": 0.0,
        "spawn_max_per_step": 40, "port_prob": 0.10, "wealth_decay": 0.0, "wealth_coastal_rate": 0.01,
    }},
    {"rate": 0.027, "round": 10, "params": {
        "spawn_prob": 0.15, "spawn_pop_threshold": 0.7, "death_base_rate": 0.080,
        "death_food_factor": 0.15, "pop_growth_rate": 0.10, "food_base_regen": 0.15,
        "food_competition": 0.05, "food_pop_drain": 0.04, "food_forest_bonus": 0.04,
        "port_survival_bonus": 0.0, "port_wealth_threshold": 0.0,
        "spawn_max_per_step": 40, "port_prob": 0.10, "wealth_decay": 0.0, "wealth_coastal_rate": 0.01,
    }},
    {"rate": 0.040, "round": 8, "params": {
        "spawn_prob": 0.10, "spawn_pop_threshold": 0.5, "death_base_rate": 0.100,
        "death_food_factor": 0.10, "pop_growth_rate": 0.10, "food_base_regen": 0.15,
        "food_competition": 0.03, "food_pop_drain": 0.06, "food_forest_bonus": 0.04,
        "port_survival_bonus": 0.0, "port_wealth_threshold": 0.0,
        "spawn_max_per_step": 40, "port_prob": 0.10, "wealth_decay": 0.0, "wealth_coastal_rate": 0.01,
    }},
    {"rate": 0.103, "round": 4, "params": {
        "spawn_prob": 0.08, "spawn_pop_threshold": 0.3, "death_base_rate": 0.030,
        "death_food_factor": 0.06, "pop_growth_rate": 0.12, "food_base_regen": 0.15,
        "food_competition": 0.03, "food_pop_drain": 0.06, "food_forest_bonus": 0.04,
        "port_survival_bonus": 0.0, "port_wealth_threshold": 0.0,
        "spawn_max_per_step": 40, "port_prob": 0.10, "wealth_decay": 0.0, "wealth_coastal_rate": 0.01,
    }},
    {"rate": 0.132, "round": 5, "params": {
        "spawn_prob": 0.08, "spawn_pop_threshold": 0.5, "death_base_rate": 0.010,
        "death_food_factor": 0.06, "pop_growth_rate": 0.08, "food_base_regen": 0.20,
        "food_competition": 0.02, "food_pop_drain": 0.08, "food_forest_bonus": 0.02,
        "port_survival_bonus": 0.02, "port_wealth_threshold": 0.0,
        "spawn_max_per_step": 40, "port_prob": 0.10, "wealth_decay": 0.0, "wealth_coastal_rate": 0.01,
    }},
    {"rate": 0.138, "round": 9, "params": {
        "spawn_prob": 0.10, "spawn_pop_threshold": 0.5, "death_base_rate": 0.020,
        "death_food_factor": 0.04, "pop_growth_rate": 0.12, "food_base_regen": 0.20,
        "food_competition": 0.03, "food_pop_drain": 0.06, "food_forest_bonus": 0.04,
        "port_survival_bonus": 0.0, "port_wealth_threshold": 0.0,
        "spawn_max_per_step": 40, "port_prob": 0.10, "wealth_decay": 0.0, "wealth_coastal_rate": 0.01,
    }},
    {"rate": 0.161, "round": 1, "params": {
        "spawn_prob": 0.08, "spawn_pop_threshold": 0.5, "death_base_rate": 0.005,
        "death_food_factor": 0.02, "pop_growth_rate": 0.10, "food_base_regen": 0.25,
        "food_competition": 0.03, "food_pop_drain": 0.04, "food_forest_bonus": 0.06,
        "port_survival_bonus": 0.02, "port_wealth_threshold": 0.0,
        "spawn_max_per_step": 40, "port_prob": 0.10, "wealth_decay": 0.0, "wealth_coastal_rate": 0.01,
    }},
    {"rate": 0.186, "round": 2, "params": {
        "spawn_prob": 0.10, "spawn_pop_threshold": 0.5, "death_base_rate": 0.015,
        "death_food_factor": 0.04, "pop_growth_rate": 0.12, "food_base_regen": 0.20,
        "food_competition": 0.03, "food_pop_drain": 0.04, "food_forest_bonus": 0.04,
        "port_survival_bonus": 0.0, "port_wealth_threshold": 0.0,
        "spawn_max_per_step": 40, "port_prob": 0.10, "wealth_decay": 0.0, "wealth_coastal_rate": 0.01,
    }},
    {"rate": 0.238, "round": 6, "params": {
        "spawn_prob": 0.12, "spawn_pop_threshold": 0.5, "death_base_rate": 0.005,
        "death_food_factor": 0.06, "pop_growth_rate": 0.10, "food_base_regen": 0.20,
        "food_competition": 0.03, "food_pop_drain": 0.06, "food_forest_bonus": 0.06,
        "port_survival_bonus": 0.0, "port_wealth_threshold": 0.0,
        "spawn_max_per_step": 40, "port_prob": 0.10, "wealth_decay": 0.0, "wealth_coastal_rate": 0.01,
    }},
]


def params_from_rate(settle_rate):
    """Interpolate SimParams from observed settle rate.
    Uses weighted blend of two nearest calibration points.
    """
    from simulator_exp import SimParams
    pts = _CALIBRATION_POINTS
    rates = [p["rate"] for p in pts]

    if settle_rate <= rates[0]:
        return SimParams(**pts[0]["params"])
    if settle_rate >= rates[-1]:
        return SimParams(**pts[-1]["params"])

    for i in range(len(rates) - 1):
        if rates[i] <= settle_rate <= rates[i + 1]:
            lo, hi = pts[i], pts[i + 1]
            t = (settle_rate - lo["rate"]) / (hi["rate"] - lo["rate"])
            blended = {}
            for k in lo["params"]:
                blended[k] = lo["params"][k] * (1 - t) + hi["params"][k] * t
            return SimParams(**blended)

    return SimParams(**pts[len(pts) // 2]["params"])


class AstarSolver:
    BASE_URL = "https://api.ainm.no"
    SIMULATE_DELAY = 0.25   # stay under 5 req/s
    SUBMIT_DELAY = 0.55     # stay under 2 req/s
    MIN_PROB = 0.001        # probability floor
    VP_MAX = 15             # max viewport dimension
    DYNAMIC_RANGE = 6       # manhattan distance considered "near settlement"

    def __init__(self, token: str, use_mc: bool = True):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })
        self.ml_models = self._load_ml_models()
        self.use_mc = use_mc
        self.round_type = "normal"  # updated by _detect_round_type

    @staticmethod
    def _load_ml_models():
        """Load pre-trained LightGBM models if available."""
        model_path = os.path.join(os.path.dirname(__file__) or ".", "ml_models.pkl")
        try:
            with open(model_path, "rb") as f:
                models = pickle.load(f)
            logger.info("Loaded %d ML models from %s", len(models), model_path)
            return models
        except Exception:
            logger.info("No ML models found at %s — using hand-crafted only", model_path)
            return None

    # ------------------------------------------------------------------
    # Round-type dependent parameters
    # ------------------------------------------------------------------
    @property
    def _ml_weight(self):
        return 0.30 if self.round_type == "normal" else 0.0

    @property
    def _wavefront_factor(self):
        return 0.03 if self.round_type == "normal" else 0.01

    @property
    def _clustering_enabled(self):
        return self.round_type == "normal"

    @property
    def _mc_weight(self):
        base = 0.40 if self.round_type == "normal" else 0.60
        return base * getattr(self, "_mc_weight_adjustment", 1.0)

    @property
    def _prior_weight(self):
        base = 0.30 if self.round_type == "normal" else 0.40
        # Absorb weight lost from MC adjustment
        mc_lost = (0.40 if self.round_type == "normal" else 0.60) * (1.0 - getattr(self, "_mc_weight_adjustment", 1.0))
        return base + mc_lost

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------
    def _get(self, path: str):
        r = self.session.get(f"{self.BASE_URL}{path}", timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict):
        r = self.session.post(f"{self.BASE_URL}{path}", json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def get_rounds(self):
        return self._get("/astar-island/rounds")

    def get_round_detail(self, round_id: str):
        return self._get(f"/astar-island/rounds/{round_id}")

    def get_budget(self):
        return self._get("/astar-island/budget")

    def simulate(self, round_id, seed_index, vx, vy, vw, vh):
        return self._post("/astar-island/simulate", {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": vx,
            "viewport_y": vy,
            "viewport_w": vw,
            "viewport_h": vh,
        })

    def submit(self, round_id, seed_index, prediction):
        return self._post("/astar-island/submit", {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        })

    # ------------------------------------------------------------------
    # Initial-state analysis
    # ------------------------------------------------------------------
    def analyse_seed(self, state: dict, W: int, H: int) -> dict:
        grid = np.array(state["grid"], dtype=int)
        settlements = state["settlements"]
        spos = [(s["x"], s["y"]) for s in settlements if s.get("alive", True)]

        # Distance to nearest settlement (vectorised)
        dist = np.full((H, W), 999.0)
        ys, xs = np.mgrid[0:H, 0:W]
        for sx, sy in spos:
            d = np.abs(xs - sx) + np.abs(ys - sy)
            dist = np.minimum(dist, d.astype(float))

        # Coastal mask (land cell adjacent to ocean)
        ocean = grid == 10
        land = ~ocean
        coastal = np.zeros((H, W), dtype=bool)
        if H > 1:
            coastal[1:, :] |= land[1:, :] & ocean[:-1, :]
            coastal[:-1, :] |= land[:-1, :] & ocean[1:, :]
        if W > 1:
            coastal[:, 1:] |= land[:, 1:] & ocean[:, :-1]
            coastal[:, :-1] |= land[:, :-1] & ocean[:, 1:]

        # Settlement density: count of settlements within radius 5 per cell
        n_nearby = np.zeros((H, W), dtype=int)
        for sx, sy in spos:
            mask = (np.abs(xs - sx) + np.abs(ys - sy)) <= 5
            n_nearby[mask] += 1

        # Local clustering: count of settlements within radius 2 per cell
        n_local = np.zeros((H, W), dtype=int)
        for sx, sy in spos:
            mask = (np.abs(xs - sx) + np.abs(ys - sy)) <= 2
            n_local[mask] += 1

        # Adjacent forest count (4-connected neighbors)
        forest = (grid == 4).astype(int)
        adj_forest = np.zeros((H, W), dtype=int)
        if H > 1:
            adj_forest[1:] += forest[:-1]
            adj_forest[:-1] += forest[1:]
        if W > 1:
            adj_forest[:, 1:] += forest[:, :-1]
            adj_forest[:, :-1] += forest[:, 1:]

        # Dynamic score per cell (higher → more likely to change)
        RNG = self.DYNAMIC_RANGE
        in_range = dist <= RNG
        not_static = (grid != 10) & (grid != 5)  # not ocean/mountain
        base = np.maximum(1, RNG - dist + 1) * not_static * in_range
        boost = np.where(np.isin(grid, [1, 2]), 10, 0) + np.where(grid == 3, 6, 0)
        forest_near = np.where((grid == 4) & (dist <= 3), 5, 0)
        dynamic = base + (boost + forest_near) * in_range

        return {
            "grid": grid,
            "settlements": settlements,
            "spos": spos,
            "dist": dist,
            "coastal": coastal,
            "dynamic": dynamic,
            "n_nearby": n_nearby,
            "n_local": n_local,
            "adj_forest": adj_forest,
        }

    # ------------------------------------------------------------------
    # Round-type detection
    # ------------------------------------------------------------------
    def _detect_round_type(self, scout_obs: list) -> str:
        """Detect whether this is a normal or die-off round from scout observations.

        Returns "normal" or "dieoff".
        """
        total_cells = 0
        total_settlements = 0
        total_ruins = 0

        for obs in scout_obs:
            grid = np.array(obs["grid"])
            total_cells += grid.size
            total_settlements += int(np.sum((grid == 1) | (grid == 2)))
            total_ruins += int(np.sum(grid == 3))

        if total_cells == 0:
            return "normal"  # no data, default

        settlement_rate = total_settlements / total_cells
        ruin_rate = total_ruins / total_cells

        logger.info("Round detection: settlement_rate=%.4f, ruin_rate=%.4f, cells=%d",
                     settlement_rate, ruin_rate, total_cells)

        if settlement_rate < 0.03:
            # Very few settlements → die-off (R3=0.23%, R8=2.2%, R10=0.8%)
            logger.info("Detected: DIE-OFF (settlement_rate < 0.03)")
            return "dieoff"
        elif settlement_rate > 0.08:
            # Lots of settlements → normal (R1-R7 = 10-20%)
            logger.info("Detected: NORMAL (settlement_rate > 0.08)")
            return "normal"
        else:
            # Ambiguous zone (0.03-0.08): check ruin ratio
            if ruin_rate > 0.02:
                logger.info("Detected: DIE-OFF (ambiguous zone, high ruin_rate)")
                return "dieoff"
            else:
                logger.info("Detected: NORMAL (ambiguous zone, conservative)")
                return "normal"

    # ------------------------------------------------------------------
    # MC prediction
    # ------------------------------------------------------------------
    def _run_mc_prediction(self, analysis: dict, sim_params: SimParams,
                           n_runs: int = 500, timeout: float = 120.0) -> Optional[np.ndarray]:
        """Run Monte Carlo simulation to predict terrain probabilities.

        Returns (H, W, 6) probability array, or None on failure/timeout.
        """
        if not self.use_mc:
            return None

        grid = analysis["grid"]
        settlements = analysis["settlements"]

        # Build settlement list with default attributes
        sim_settlements = []
        for s in settlements:
            if not s.get("alive", True):
                continue
            sim_settlements.append({
                "x": s["x"],
                "y": s["y"],
                "population": s.get("population", 0.5),
                "food": s.get("food", 0.5),
                "wealth": s.get("wealth", 0.0),
                "defense": s.get("defense", 0.2),
                "has_port": s.get("has_port", False),
                "owner_id": s.get("owner_id", 0),
                "alive": True,
            })

        try:
            t0 = time.time()
            sim = AstarSimulator(grid, sim_settlements, params=sim_params, seed=42)
            probs = sim.monte_carlo(n_runs=n_runs, n_steps=50, n_workers=0)
            elapsed = time.time() - t0
            logger.info("MC prediction: %d runs in %.1fs", n_runs, elapsed)
            if elapsed > timeout:
                logger.warning("MC prediction took %.1fs (timeout=%.1fs)", elapsed, timeout)
            return probs
        except Exception as e:
            logger.error("MC prediction failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Query planning
    # ------------------------------------------------------------------
    SCOUT_QUERIES_PER_SEED = 2  # initial recon per seed
    REPEAT_QUERIES_PER_SEED = 3  # repeated viewport sampling per seed

    def plan_scout_queries(
        self, analyses: list, n_seeds: int, W: int, H: int
    ) -> List[dict]:
        plan: List[dict] = []
        for si in range(n_seeds):
            vps = self._plan_viewports(analyses[si], self.SCOUT_QUERIES_PER_SEED, W, H)
            for vp in vps:
                plan.append({"seed": si, "x": vp[0], "y": vp[1], "w": vp[2], "h": vp[3]})
        return plan

    def plan_deep_queries(
        self, analyses: list, scout_obs: list, budget: int,
        n_seeds: int, W: int, H: int
    ) -> List[dict]:
        """Allocate remaining budget across seeds based on scout observations."""
        seed_dynamism = [0.0] * n_seeds
        seed_coverage = [0.0] * n_seeds
        for obs in scout_obs:
            si = obs["seed"]
            grid = np.array(obs["grid"])
            n_settle = np.sum((grid == 1) | (grid == 2))
            n_ruin = np.sum(grid == 3)
            seed_dynamism[si] += float(n_settle * 2 + n_ruin * 5)
            seed_coverage[si] += grid.size

        for si in range(n_seeds):
            if seed_coverage[si] == 0:
                seed_dynamism[si] = float(analyses[si]["dynamic"].sum())

        total = sum(seed_dynamism) or 1.0

        alloc = [0] * n_seeds
        for si in range(n_seeds):
            share = seed_dynamism[si] / total
            alloc[si] = max(2, int(budget * share)) if seed_dynamism[si] > 0 else 0

        while sum(alloc) > budget:
            biggest = max(range(n_seeds), key=lambda i: alloc[i])
            alloc[biggest] -= 1

        ranked = sorted(range(n_seeds), key=lambda i: -seed_dynamism[i])
        while sum(alloc) < budget:
            for si in ranked:
                if sum(alloc) >= budget:
                    break
                alloc[si] += 1

        logger.info("Deep query allocation: %s (dynamism: %s)",
                     alloc, [f"{d:.0f}" for d in seed_dynamism])

        plan: List[dict] = []
        for si in range(n_seeds):
            if alloc[si] > 0:
                vps = self._plan_viewports(analyses[si], alloc[si], W, H)
                for vp in vps:
                    plan.append({"seed": si, "x": vp[0], "y": vp[1], "w": vp[2], "h": vp[3]})
        return plan

    def _plan_viewports(
        self, analysis: dict, n_queries: int, W: int, H: int
    ) -> List[Tuple[int, int, int, int]]:
        """Greedily place viewports to maximise observation of dynamic cells."""
        VP = self.VP_MAX
        dyn = analysis["dynamic"]
        coverage = np.zeros((H, W), dtype=int)

        max_sx = max(0, W - 5)
        max_sy = max(0, H - 5)
        pos_x = sorted(set(list(range(0, max_sx + 1, 2)) + [max_sx]))
        pos_y = sorted(set(list(range(0, max_sy + 1, 2)) + [max_sy]))

        viewports: List[Tuple[int, int, int, int]] = []
        for _ in range(n_queries):
            best, best_s = None, -1.0
            for vy in pos_y:
                for vx in pos_x:
                    vw = min(VP, W - vx)
                    vh = min(VP, H - vy)
                    if vw < 5 or vh < 5:
                        continue
                    rd = dyn[vy : vy + vh, vx : vx + vw]
                    rc = coverage[vy : vy + vh, vx : vx + vw]
                    s = float(
                        (rd * (rc == 0)).sum() * 10
                        + (rd * (rc == 1)).sum() * 4
                        + (rd * (rc >= 2)).sum() * 1
                    )
                    if s > best_s:
                        best_s = s
                        best = (vx, vy, vw, vh)
            if best is None or best_s <= 0:
                break
            vx, vy, vw, vh = best
            coverage[vy : vy + vh, vx : vx + vw] += 1
            viewports.append(best)
        return viewports

    # ------------------------------------------------------------------
    # Repeated viewport sampling — empirical probability estimation
    # ------------------------------------------------------------------
    def plan_repeat_queries(
        self, analyses: list, scout_obs: list, budget: int,
        n_seeds: int, W: int, H: int
    ) -> List[dict]:
        """Plan repeated observations of the most dynamic viewport per seed.

        Each seed gets REPEAT_QUERIES_PER_SEED repeats of the same viewport,
        targeting the area with highest settlement activity (from scout obs).
        Returns query plan consuming at most `budget` queries.
        """
        repeats_per_seed = min(self.REPEAT_QUERIES_PER_SEED, budget // max(n_seeds, 1))
        if repeats_per_seed < 2:
            return []  # not enough budget for meaningful repeats

        plan: List[dict] = []

        for si in range(n_seeds):
            # Find best viewport from scout observations for this seed
            best_vp = None
            best_activity = -1

            for obs in scout_obs:
                if obs["seed"] != si:
                    continue
                grid = np.array(obs["grid"])
                activity = int(np.sum((grid == 1) | (grid == 2) | (grid == 3)))
                if activity > best_activity:
                    best_activity = activity
                    vp = obs["viewport"]
                    best_vp = (vp["x"], vp["y"], vp["w"], vp["h"])

            if best_vp is None:
                # No scout obs for this seed — use most dynamic area
                vps = self._plan_viewports(analyses[si], 1, W, H)
                if vps:
                    best_vp = vps[0]
                else:
                    continue

            for _ in range(repeats_per_seed):
                if len(plan) >= budget:
                    break
                plan.append({
                    "seed": si,
                    "x": best_vp[0], "y": best_vp[1],
                    "w": best_vp[2], "h": best_vp[3],
                })

        return plan[:budget]

    def build_empirical_distributions(
        self, observations: list, initial_states: list, W: int, H: int
    ) -> Optional[np.ndarray]:
        """Build per-cell empirical probability from repeated viewport observations.

        Groups observations by (seed, viewport position). For cells observed
        multiple times, computes frequency-based probability distribution.
        Returns (n_seeds, H, W, 6) array where cells with <2 observations
        are set to None (masked with -1).
        """
        from collections import Counter

        n_seeds = len(initial_states)
        # Count observations per (seed, y, x, class)
        cell_counts = {}  # (seed, y, x) → Counter of terrain classes
        cell_obs_count = {}  # (seed, y, x) → number of observations

        for obs in observations:
            si = obs["seed"]
            vp = obs["viewport"]
            for dy, row in enumerate(obs["grid"]):
                for dx, val in enumerate(row):
                    y = vp["y"] + dy
                    x = vp["x"] + dx
                    if 0 <= y < H and 0 <= x < W:
                        key = (si, y, x)
                        cls = TERRAIN_TO_CLASS.get(val, 0)
                        if key not in cell_counts:
                            cell_counts[key] = Counter()
                            cell_obs_count[key] = 0
                        cell_counts[key][cls] += 1
                        cell_obs_count[key] += 1

        # Only use cells with multiple observations
        multi_obs = {k: v for k, v in cell_counts.items() if cell_obs_count[k] >= 2}
        if not multi_obs:
            return None

        # Build probability array: -1 means no data
        empirical = np.full((n_seeds, H, W, 6), -1.0)
        for (si, y, x), counts in multi_obs.items():
            total = sum(counts.values())
            probs = np.zeros(6)
            for cls, cnt in counts.items():
                probs[cls] = cnt / total
            empirical[si, y, x] = probs

        # Store observation counts per seed for scaled blending
        self._empirical_counts = {}
        for si in range(n_seeds):
            counts_grid = np.zeros((H, W), dtype=np.int32)
            for (s, y, x), cnt in cell_obs_count.items():
                if s == si:
                    counts_grid[y, x] = cnt
            self._empirical_counts[si] = counts_grid

        n_cells = len(multi_obs)
        n_varied = sum(1 for k, v in multi_obs.items() if len(v) > 1)
        logger.info("Empirical distributions: %d cells with 2+ obs (%d varied)", n_cells, n_varied)
        return empirical

    # ------------------------------------------------------------------
    # Priors — both normal and die-off
    # ------------------------------------------------------------------
    NORMAL_PRIORS = {
        # Ocean — always stays empty
        (10, False): np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (10, True):  np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # Mountain — always stays mountain
        (5, False): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        (5, True):  np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        # Settlement (d=0)
        (1, False): np.array([0.4395, 0.3293, 0.0000, 0.0283, 0.2029, 0.0000]),
        (1, True):  np.array([0.4923, 0.0861, 0.1850, 0.0174, 0.2191, 0.0000]),
        # Port (d=0)
        (2, True):  np.array([0.4593, 0.0999, 0.1965, 0.0240, 0.2204, 0.0000]),
        (2, False): np.array([0.4593, 0.0999, 0.1965, 0.0240, 0.2204, 0.0000]),
        # Plains by (dist_bucket, coastal)
        (11, "d=1-2", False): np.array([0.7220, 0.2048, 0.0000, 0.0186, 0.0546, 0.0000]),
        (11, "d=1-2", True):  np.array([0.7243, 0.0926, 0.1243, 0.0162, 0.0427, 0.0000]),
        (11, "d=3-4", False): np.array([0.8140, 0.1365, 0.0000, 0.0140, 0.0354, 0.0000]),
        (11, "d=3-4", True):  np.array([0.8160, 0.0710, 0.0754, 0.0112, 0.0264, 0.0000]),
        (11, "d=5-7", False): np.array([0.9036, 0.0744, 0.0000, 0.0081, 0.0139, 0.0000]),
        (11, "d=5-7", True):  np.array([0.9010, 0.0451, 0.0361, 0.0065, 0.0112, 0.0000]),
        (11, "d=8+",  False): np.array([0.9725, 0.0237, 0.0000, 0.0017, 0.0021, 0.0000]),
        (11, "d=8+",  True):  np.array([0.9696, 0.0173, 0.0094, 0.0018, 0.0019, 0.0000]),
        # Forest by (dist_bucket, coastal)
        (4, "d=1-2", False): np.array([0.1229, 0.2144, 0.0000, 0.0193, 0.6434, 0.0000]),
        (4, "d=1-2", True):  np.array([0.1029, 0.1046, 0.1434, 0.0178, 0.6313, 0.0000]),
        (4, "d=3-4", False): np.array([0.0780, 0.1388, 0.0000, 0.0147, 0.7685, 0.0000]),
        (4, "d=3-4", True):  np.array([0.0561, 0.0699, 0.0737, 0.0110, 0.7893, 0.0000]),
        (4, "d=5-7", False): np.array([0.0319, 0.0790, 0.0000, 0.0082, 0.8809, 0.0000]),
        (4, "d=5-7", True):  np.array([0.0217, 0.0429, 0.0359, 0.0056, 0.8939, 0.0000]),
        (4, "d=8+",  False): np.array([0.0064, 0.0284, 0.0000, 0.0025, 0.9627, 0.0000]),
        (4, "d=8+",  True):  np.array([0.0060, 0.0149, 0.0085, 0.0018, 0.9689, 0.0000]),
    }

    NORMAL_PRIORS_4D = {
        (11, "d=1-2", False, "ld=2"): np.array([0.6532, 0.2609, 0.0000, 0.0227, 0.0632, 0.0000]),
        (11, "d=1-2", False, "ld=3+"): np.array([0.6970, 0.2727, 0.0000, 0.0000, 0.0303, 0.0000]),
        (11, "d=1-2", True, "ld=2"):  np.array([0.7442, 0.1395, 0.0930, 0.0233, 0.0000, 0.0000]),
        (4, "d=1-2", False, "ld=2"): np.array([0.1429, 0.2555, 0.0000, 0.0330, 0.5687, 0.0000]),
        (4, "d=1-2", False, "ld=3+"): np.array([0.1818, 0.3636, 0.0000, 0.0909, 0.3636, 0.0000]),
    }

    DIEOFF_PRIORS = {
        # Ocean — always stays empty
        (10, False): np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (10, True):  np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # Mountain — always stays mountain
        (5, False): np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        (5, True):  np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        # Settlement (d=0) — die-off: only 4.5% survive (vs 33% general)
        (1, False): np.array([0.6459, 0.0455, 0.0000, 0.0075, 0.3011, 0.0000]),
        (1, True):  np.array([0.6763, 0.0073, 0.0140, 0.0023, 0.3000, 0.0000]),
        # Port (d=0) — die-off
        (2, True):  np.array([0.6689, 0.0086, 0.0128, 0.0047, 0.3050, 0.0000]),
        (2, False): np.array([0.6689, 0.0086, 0.0128, 0.0047, 0.3050, 0.0000]),
        # Plains — die-off: much less expansion
        (11, "d=1-2", False): np.array([0.9346, 0.0227, 0.0000, 0.0039, 0.0388, 0.0000]),
        (11, "d=1-2", True):  np.array([0.9536, 0.0056, 0.0062, 0.0026, 0.0320, 0.0000]),
        (11, "d=3-4", False): np.array([0.9637, 0.0148, 0.0000, 0.0023, 0.0192, 0.0000]),
        (11, "d=3-4", True):  np.array([0.9723, 0.0039, 0.0045, 0.0019, 0.0174, 0.0000]),
        (11, "d=5-7", False): np.array([0.9843, 0.0064, 0.0000, 0.0010, 0.0083, 0.0000]),
        (11, "d=5-7", True):  np.array([0.9857, 0.0025, 0.0025, 0.0009, 0.0085, 0.0000]),
        (11, "d=8+",  False): np.array([0.9983, 0.0008, 0.0000, 0.0002, 0.0007, 0.0000]),
        (11, "d=8+",  True):  np.array([0.9981, 0.0004, 0.0004, 0.0002, 0.0009, 0.0000]),
        # Forest — die-off
        (4, "d=1-2", False): np.array([0.0857, 0.0237, 0.0000, 0.0039, 0.8868, 0.0000]),
        (4, "d=1-2", True):  np.array([0.0768, 0.0065, 0.0054, 0.0034, 0.9078, 0.0000]),
        (4, "d=3-4", False): np.array([0.0379, 0.0148, 0.0000, 0.0022, 0.9451, 0.0000]),
        (4, "d=3-4", True):  np.array([0.0350, 0.0038, 0.0044, 0.0016, 0.9551, 0.0000]),
        (4, "d=5-7", False): np.array([0.0170, 0.0065, 0.0000, 0.0010, 0.9754, 0.0000]),
        (4, "d=5-7", True):  np.array([0.0178, 0.0026, 0.0020, 0.0009, 0.9766, 0.0000]),
        (4, "d=8+",  False): np.array([0.0020, 0.0013, 0.0000, 0.0003, 0.9964, 0.0000]),
        (4, "d=8+",  True):  np.array([0.0024, 0.0004, 0.0007, 0.0001, 0.9964, 0.0000]),
    }

    # No 4D priors in die-off mode — clustering doesn't help when everything dies
    DIEOFF_PRIORS_4D = {}

    @staticmethod
    def _dist_bucket(d: float) -> str:
        if d < 1:
            return "d=0"
        elif d < 3:
            return "d=1-2"
        elif d < 5:
            return "d=3-4"
        elif d < 8:
            return "d=5-7"
        else:
            return "d=8+"

    def _get_priors(self):
        """Return the prior tables for the current round type."""
        if self.round_type == "dieoff":
            return self.DIEOFF_PRIORS, self.DIEOFF_PRIORS_4D
        return self.NORMAL_PRIORS, self.NORMAL_PRIORS_4D

    def _prior(self, terrain: int, dist: float, is_coastal: bool,
               n_nearby: int = 0, adj_forest: int = 0,
               n_local: int = 0) -> np.ndarray:
        """Dirichlet alpha prior [empty, settlement, port, ruin, forest, mountain]."""
        STRENGTH_STATIC = 100.0
        STRENGTH_DYNAMIC = 8.0

        priors, priors_4d = self._get_priors()
        bucket = self._dist_bucket(dist)

        # Static terrains
        if terrain == 10:
            return np.array([STRENGTH_STATIC, 0.01, 0.01, 0.01, 0.01, 0.01])
        if terrain == 5:
            return np.array([0.01, 0.01, 0.01, 0.01, 0.01, STRENGTH_STATIC])

        # Settlement / Port
        if terrain in (1, 2):
            key = (terrain, is_coastal)
            base = priors.get(key, priors.get((terrain, not is_coastal))).copy()
            if self._clustering_enabled:
                cluster_boost = 1.0 + 0.15 * min(n_local, 4)
                base[1] *= cluster_boost
                base[2] *= cluster_boost
                base[0] *= max(0.4, 1.0 / cluster_boost)
                forest_factor = 1.0 + 0.06 * min(adj_forest, 3)
                base[1] *= forest_factor
                base[0] *= (2.0 - forest_factor)
            base = base / base.sum()
            return np.maximum(base * STRENGTH_DYNAMIC, 0.01)

        # Plains / Forest
        if terrain in (4, 11, 0):
            lookup_terrain = 11 if terrain in (0, 11) else 4

            # 4D lookup for ld >= 2
            if n_local >= 2 and priors_4d:
                ld_str = "ld=2" if n_local == 2 else "ld=3+"
                key_4d = (lookup_terrain, bucket, is_coastal, ld_str)
                if key_4d in priors_4d:
                    base = priors_4d[key_4d].copy()
                    return np.maximum(base * STRENGTH_DYNAMIC, 0.01)

            # 3D lookup
            key = (lookup_terrain, bucket, is_coastal)
            if key not in priors:
                key = (lookup_terrain, bucket, not is_coastal)
            if key not in priors:
                key = (lookup_terrain, "d=8+", is_coastal)
            if key not in priors:
                key = (lookup_terrain, "d=8+", False)
            base = priors[key].copy()

            # Clustering boost (normal only)
            if self._clustering_enabled and bucket in ("d=1-2", "d=3-4") and n_local > 0:
                cluster_boost = 1.0 + 0.35 * min(n_local, 4)
                base[1] *= cluster_boost
                base[2] *= cluster_boost
                base[0] *= max(0.3, 1.0 / cluster_boost)
                base = base / base.sum()
            return np.maximum(base * STRENGTH_DYNAMIC, 0.01)

        # Ruin
        if terrain == 3:
            return np.array([0.4, 0.15, 0.05, 0.15, 0.25, 0.01]) * STRENGTH_DYNAMIC

        # Unknown fallback
        return np.ones(6) / 6 * STRENGTH_DYNAMIC

    # ------------------------------------------------------------------
    # Cross-seed learning
    # ------------------------------------------------------------------
    def learn_transitions(
        self, observations: list, initial_states: list
    ) -> dict:
        fine_counts: Dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(6))
        coarse_counts: Dict[int, np.ndarray] = defaultdict(lambda: np.zeros(6))

        analyses = {}
        for si, state in enumerate(initial_states):
            W = len(state["grid"][0]) if state["grid"] else 0
            H = len(state["grid"])
            if (si, "grid") not in analyses and H > 0:
                analyses[si] = self.analyse_seed(state, W, H)

        for obs in observations:
            si = obs["seed"]
            if si not in analyses:
                continue
            a = analyses[si]
            init_grid = a["grid"]
            dist_arr = a["dist"]
            coastal_arr = a["coastal"]
            vp = obs["viewport"]
            for dy, row in enumerate(obs["grid"]):
                for dx, val in enumerate(row):
                    y, x = vp["y"] + dy, vp["x"] + dx
                    if 0 <= y < init_grid.shape[0] and 0 <= x < init_grid.shape[1]:
                        tv = int(init_grid[y, x])
                        cls = TERRAIN_TO_CLASS.get(val, 0)
                        bucket = self._dist_bucket(float(dist_arr[y, x]))
                        coast = bool(coastal_arr[y, x])
                        fine_counts[(tv, bucket, coast)][cls] += 1
                        coarse_counts[tv][cls] += 1

        result = {"fine": {}, "coarse": {}}
        for key, c in fine_counts.items():
            t = c.sum()
            if t >= 3:
                result["fine"][key] = c / t
        for tv, c in coarse_counts.items():
            t = c.sum()
            if t > 0:
                result["coarse"][tv] = c / t
        return result

    # ------------------------------------------------------------------
    # Per-round parameter estimation
    # ------------------------------------------------------------------
    def estimate_round_params(self, transitions: dict) -> dict:
        fine = transitions.get("fine", {}) if isinstance(transitions, dict) else {}
        if not fine:
            return {}

        if self.round_type == "dieoff":
            hist_plains_settle = 0.023
            hist_forest_settle = 0.024
            hist_settle_survive = 0.045
        else:
            hist_plains_settle = 0.205
            hist_forest_settle = 0.214
            hist_settle_survive = 0.329

        obs_plains = fine.get((11, "d=1-2", False))
        obs_forest = fine.get((4, "d=1-2", False))
        obs_settle = fine.get((1, "d=0", False))
        coarse = transitions.get("coarse", {})
        if obs_settle is None:
            obs_settle = coarse.get(1)

        params = {}

        expansion_samples = []
        if obs_plains is not None:
            expansion_samples.append(obs_plains[1] / max(hist_plains_settle, 0.01))
        if obs_forest is not None:
            expansion_samples.append(obs_forest[1] / max(hist_forest_settle, 0.01))

        if expansion_samples:
            params["expansion_ratio"] = float(np.mean(expansion_samples))
        else:
            params["expansion_ratio"] = 1.0

        if obs_settle is not None:
            params["survival_ratio"] = float(obs_settle[1] / max(hist_settle_survive, 0.01))
        else:
            params["survival_ratio"] = 1.0

        return params

    # ------------------------------------------------------------------
    # Prediction building
    # ------------------------------------------------------------------
    def build_prediction(
        self,
        seed_idx: int,
        analysis: dict,
        observations: list,
        transitions: Dict[int, np.ndarray],
        W: int,
        H: int,
        mc_pred: Optional[np.ndarray] = None,
        empirical: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        grid = analysis["grid"]
        dist = analysis["dist"]
        coastal = analysis["coastal"]
        dynamic = analysis["dynamic"]
        n_nearby = analysis.get("n_nearby", np.zeros_like(grid))
        n_local = analysis.get("n_local", np.zeros_like(grid))
        adj_forest = analysis.get("adj_forest", np.zeros_like(grid))

        round_params = self.estimate_round_params(transitions)
        exp_ratio = round_params.get("expansion_ratio", 1.0)
        surv_ratio = round_params.get("survival_ratio", 1.0)

        fine_trans = transitions.get("fine", {}) if isinstance(transitions, dict) else {}
        coarse_trans = transitions.get("coarse", {}) if isinstance(transitions, dict) else transitions

        # Build prior prediction
        prior_pred = np.zeros((H, W, 6))
        for y in range(H):
            for x in range(W):
                tv = int(grid[y, x])
                d = float(dist[y, x])
                c = bool(coastal[y, x])
                alphas = self._prior(tv, d, c, int(n_nearby[y, x]), int(adj_forest[y, x]),
                                     int(n_local[y, x]))

                # Apply round-specific parameter scaling
                if exp_ratio != 1.0:
                    if tv in (11, 0, 4):
                        alphas[1] *= exp_ratio
                        alphas[2] *= exp_ratio
                        alphas[3] *= exp_ratio
                    elif tv in (1, 2):
                        alphas[1] *= surv_ratio
                        alphas[2] *= surv_ratio

                if dynamic[y, x] > 0:
                    bucket = self._dist_bucket(d)
                    trans = fine_trans.get((tv, bucket, c))
                    if trans is None:
                        trans = fine_trans.get((tv, bucket, not c))
                    if trans is None and isinstance(coarse_trans, dict):
                        trans = coarse_trans.get(tv)
                    if trans is not None:
                        weight = min(4.0, dynamic[y, x] / 3.0)
                        alphas = alphas + trans * weight

                prior_pred[y, x] = alphas / alphas.sum()

        # Blend: MC + prior + ML for dynamic cells; prior only for static
        mc_w = self._mc_weight
        prior_w = self._prior_weight
        ml_w = self._ml_weight

        # Get ML prediction if available
        ml_pred = None
        if self.ml_models is not None and ml_w > 0:
            try:
                from train_model import compute_features
                settlements_list = analysis.get("settlements", [])
                features = compute_features(grid, settlements_list, W, H)
                ml_pred = np.column_stack([m.predict(features) for m in self.ml_models])
                ml_pred = np.maximum(ml_pred, self.MIN_PROB)
                ml_pred = ml_pred / ml_pred.sum(axis=-1, keepdims=True)
                ml_pred = ml_pred.reshape(H, W, 6)
            except Exception as e:
                logger.warning("ML prediction failed: %s", e)
                ml_pred = None

        # Temperature sharpening on MC — concentrate probability mass
        MC_TEMPERATURE = 1.0  # >1 = sharper, 1.0 = no change. Tested: sharpening hurts MC-only.
        if mc_pred is not None and MC_TEMPERATURE != 1.0:
            mc_sharp = np.power(np.maximum(mc_pred, 1e-10), MC_TEMPERATURE)
            mc_sharp /= mc_sharp.sum(axis=-1, keepdims=True)
        else:
            mc_sharp = mc_pred

        # Build blended prediction
        if mc_sharp is not None:
            # Dynamic cells: blend MC + prior + ML
            # Static cells (ocean, mountain, far from settlements): prior only
            is_dynamic = dist <= self.DYNAMIC_RANGE
            pred = np.zeros((H, W, 6))

            # Prior for all cells
            pred[:] = prior_pred

            # Blend for dynamic cells
            dynamic_mask = is_dynamic & (grid != 10) & (grid != 5)
            if dynamic_mask.any():
                # Normalize weights (ML may be 0)
                effective_ml_w = ml_w if ml_pred is not None else 0.0
                total_w = mc_w + prior_w + effective_ml_w
                norm_mc = mc_w / total_w
                norm_prior = prior_w / total_w
                norm_ml = effective_ml_w / total_w

                blend = norm_mc * mc_sharp + norm_prior * prior_pred
                if ml_pred is not None and norm_ml > 0:
                    blend += norm_ml * ml_pred

                # Apply blend only to dynamic cells
                mask_3d = dynamic_mask[:, :, np.newaxis]
                pred = np.where(mask_3d, blend, pred)
        else:
            # No MC: fallback to prior + ML (old behavior)
            pred = prior_pred
            if ml_pred is not None and ml_w > 0:
                pred = (1 - ml_w) * pred + ml_w * ml_pred

        # Wavefront smoothing
        settle_prob = pred[:, :, 1].copy()
        boost = np.zeros_like(settle_prob)
        if H > 1:
            boost[1:] += settle_prob[:-1]
            boost[:-1] += settle_prob[1:]
        if W > 1:
            boost[:, 1:] += settle_prob[:, :-1]
            boost[:, :-1] += settle_prob[:, 1:]
        boost *= self._wavefront_factor
        not_static = (grid != 10) & (grid != 5)
        pred[:, :, 1] += boost * not_static
        pred[:, :, 0] -= boost * not_static * 0.7
        pred[:, :, 4] -= boost * not_static * 0.3

        # Empirical distribution blending: when we have REPEATED observations
        # of the same cell (2+), we can estimate the true probability distribution.
        # This is safe because it's not a single-observation hard override —
        # it's a frequency-based estimate from multiple samples.
        # Single observations are still NEVER used as overlays (R10 lesson).
        if empirical is not None:
            emp_seed = empirical[seed_idx]  # (H, W, 6)
            has_empirical = emp_seed[:, :, 0] >= 0  # cells with data
            if has_empirical.any():
                # Scale empirical weight by observation count
                # More observations → sharper data → more trust
                # emp_seed values are normalized probs; we need obs counts
                # Use a base weight that increases with confidence
                emp_counts = getattr(self, '_empirical_counts', {}).get(seed_idx)
                if emp_counts is not None:
                    # emp_counts is (H, W) with number of observations per cell
                    emp_weight_map = np.where(emp_counts >= 8, 0.70,
                                    np.where(emp_counts >= 5, 0.55,
                                    np.where(emp_counts >= 3, 0.40, 0.20)))
                    emp_weight_3d = emp_weight_map[:, :, np.newaxis]
                    mask_3d = has_empirical[:, :, np.newaxis]
                    blended = (1 - emp_weight_3d) * pred + emp_weight_3d * emp_seed
                    pred = np.where(mask_3d, blended, pred)
                else:
                    # Fallback: flat 40% weight
                    emp_weight = 0.40
                    mask_3d = has_empirical[:, :, np.newaxis]
                    blended = (1 - emp_weight) * pred + emp_weight * emp_seed
                    pred = np.where(mask_3d, blended, pred)

        # Floor and renormalise
        pred = np.maximum(pred, self.MIN_PROB)
        pred /= pred.sum(axis=-1, keepdims=True)
        return pred

    # ------------------------------------------------------------------
    # Observation persistence
    # ------------------------------------------------------------------
    OBS_DIR = os.path.join(os.path.dirname(__file__) or ".", "obs_cache")

    def _save_observations(self, round_id: str, observations: list):
        os.makedirs(self.OBS_DIR, exist_ok=True)
        path = os.path.join(self.OBS_DIR, f"{round_id}.json")
        import json
        with open(path, "w") as f:
            json.dump(observations, f)
        logger.info("Saved %d observations to %s", len(observations), path)

    def _load_observations(self, round_id: str) -> list:
        import json
        path = os.path.join(self.OBS_DIR, f"{round_id}.json")
        try:
            with open(path) as f:
                obs = json.load(f)
            logger.info("Loaded %d cached observations from %s", len(obs), path)
            return obs
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def solve_round(self, round_id: Optional[str] = None) -> dict:
        from sim_params import NORMAL_PARAMS, DIEOFF_PARAMS

        # 1. Find active round
        if round_id is None:
            rounds = self.get_rounds()
            active = next((r for r in rounds if r["status"] == "active"), None)
            if not active:
                return {"error": "No active round found"}
            round_id = active["id"]
            logger.info("Active round %s (#%s)", round_id[:8], active.get("round_number"))

        # 2. Fetch round details
        detail = self.get_round_detail(round_id)
        W = detail["map_width"]
        H = detail["map_height"]
        n_seeds = detail["seeds_count"]
        states = detail["initial_states"]
        logger.info("Map %dx%d, %d seeds", W, H, n_seeds)

        # 3. Analyse all seeds
        analyses = [self.analyse_seed(s, W, H) for s in states]

        # 4. Safety net: MC prediction with normal params → submit immediately
        logger.info("Safety net: running MC with normal params...")
        mc_preds_safety = {}
        for si in range(n_seeds):
            mc_pred = self._run_mc_prediction(analyses[si], NORMAL_PARAMS, n_runs=500)
            mc_preds_safety[si] = mc_pred
            pred = self.build_prediction(si, analyses[si], [], {}, W, H, mc_pred=mc_pred)
            try:
                self.submit(round_id, si, pred.tolist())
                time.sleep(self.SUBMIT_DELAY)
            except Exception as e:
                logger.warning("Safety submit seed %d failed: %s", si, e)
        logger.info("Safety net submitted")

        # 5. Check remaining budget
        try:
            budget_info = self.get_budget()
            remaining = budget_info["queries_max"] - budget_info["queries_used"]
        except Exception:
            remaining = 50
        logger.info("Query budget remaining: %d", remaining)

        cached_obs = self._load_observations(round_id)

        if remaining <= 0:
            if cached_obs:
                logger.info("No budget but have %d cached obs — re-submitting", len(cached_obs))
                self.round_type = self._detect_round_type(cached_obs)
                total_cells = sum(np.array(obs["grid"]).size for obs in cached_obs)
                total_settle = sum(int(np.sum((np.array(obs["grid"]) == 1) | (np.array(obs["grid"]) == 2))) for obs in cached_obs)
                self._observed_settle_rate = total_settle / max(total_cells, 1)
                sim_params = params_from_rate(self._observed_settle_rate)
                logger.info("Interpolated params for rate=%.4f", self._observed_settle_rate)
                transitions = self.learn_transitions(cached_obs, states)
                empirical = self.build_empirical_distributions(cached_obs, states, W, H)
                for si in range(n_seeds):
                    mc_pred = self._run_mc_prediction(analyses[si], sim_params, n_runs=500)
                    pred = self.build_prediction(si, analyses[si], cached_obs, transitions, W, H,
                                                 mc_pred=mc_pred, empirical=empirical)
                    try:
                        self.submit(round_id, si, pred.tolist())
                        time.sleep(self.SUBMIT_DELAY)
                    except Exception as e:
                        logger.warning("Cached submit seed %d failed: %s", si, e)
            return {"status": "completed_prior_only", "round_id": round_id}

        # ============================================================
        # PHASE 1: Scout queries — 2 per seed to learn round dynamics
        # ============================================================
        scout_budget = min(remaining, self.SCOUT_QUERIES_PER_SEED * n_seeds)
        scout_plan = self.plan_scout_queries(analyses, n_seeds, W, H)
        logger.info("Scout phase: %d queries planned", len(scout_plan))

        scout_obs: list = list(cached_obs)
        for i, q in enumerate(scout_plan):
            if i >= scout_budget:
                break
            try:
                result = self.simulate(
                    round_id, q["seed"], q["x"], q["y"], q["w"], q["h"]
                )
                scout_obs.append({
                    "seed": q["seed"],
                    "viewport": result["viewport"],
                    "grid": result["grid"],
                    "settlements": result.get("settlements", []),
                })
                logger.info(
                    "Scout %d/%d  seed=%d vp=(%d,%d %dx%d)",
                    i + 1, len(scout_plan), q["seed"],
                    q["x"], q["y"], q["w"], q["h"],
                )
                time.sleep(self.SIMULATE_DELAY)
            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code == 429:
                    logger.warning("Rate limit hit during scout phase")
                    break
                logger.error("Scout HTTP %d: %s", code, e)
                break
            except Exception as e:
                logger.error("Scout query failed: %s", e)
                break

        self._save_observations(round_id, scout_obs)

        # 6. Auto-detect round type + interpolate params from settle rate
        self.round_type = self._detect_round_type(scout_obs)
        # Measure settle rate for param interpolation
        total_cells = sum(np.array(obs["grid"]).size for obs in scout_obs)
        total_settle = sum(int(np.sum((np.array(obs["grid"]) == 1) | (np.array(obs["grid"]) == 2))) for obs in scout_obs)
        self._observed_settle_rate = total_settle / max(total_cells, 1)
        sim_params = params_from_rate(self._observed_settle_rate)
        logger.info("Round type: %s, settle_rate=%.4f → interpolated params (sp=%.3f db=%.4f fbr=%.3f)",
                     self.round_type, self._observed_settle_rate,
                     sim_params.spawn_prob, sim_params.death_base_rate, sim_params.food_base_regen)

        # Submit MC-informed mid-round predictions
        scout_transitions = self.learn_transitions(scout_obs, states)
        logger.info("Scout done: %d obs, submitting mid-round MC predictions", len(scout_obs))

        # Adaptive MC weight: compare MC settle rate with observed
        self._mc_weight_adjustment = 1.0
        for si in range(n_seeds):
            mc_pred = self._run_mc_prediction(analyses[si], sim_params, n_runs=500)
            if mc_pred is not None and si == 0:
                mc_settle_rate = float(mc_pred[:, :, 1].mean() + mc_pred[:, :, 5].mean())
                deviation = abs(mc_settle_rate - self._observed_settle_rate) / max(self._observed_settle_rate, 0.01)
                if deviation > 0.30:
                    self._mc_weight_adjustment = 0.5
                    logger.info("MC deviates %.0f%% from obs (MC=%.3f obs=%.3f) → halving MC weight",
                                deviation * 100, mc_settle_rate, self._observed_settle_rate)
                else:
                    logger.info("MC matches obs (MC=%.3f obs=%.3f, dev=%.0f%%)",
                                mc_settle_rate, self._observed_settle_rate, deviation * 100)
            pred = self.build_prediction(si, analyses[si], scout_obs, scout_transitions, W, H, mc_pred=mc_pred)
            try:
                self.submit(round_id, si, pred.tolist())
                time.sleep(self.SUBMIT_DELAY)
            except Exception as e:
                logger.warning("Mid-round submit seed %d failed: %s", si, e)

        # ============================================================
        # PHASE 1b: Repeated viewport sampling — empirical distributions
        # ============================================================
        try:
            budget_info = self.get_budget()
            remaining = budget_info["queries_max"] - budget_info["queries_used"]
        except Exception:
            remaining = max(0, remaining - len(scout_obs))

        if remaining <= 0:
            return {
                "status": "completed_scout_only",
                "round_id": round_id,
                "round_type": self.round_type,
                "queries_executed": len(scout_obs),
            }

        repeat_budget = min(remaining // 3, self.REPEAT_QUERIES_PER_SEED * n_seeds)
        repeat_plan = self.plan_repeat_queries(
            analyses, scout_obs, repeat_budget, n_seeds, W, H
        )
        logger.info("Repeat phase: %d queries planned", len(repeat_plan))

        repeat_obs = list(scout_obs)
        for i, q in enumerate(repeat_plan):
            try:
                result = self.simulate(
                    round_id, q["seed"], q["x"], q["y"], q["w"], q["h"]
                )
                repeat_obs.append({
                    "seed": q["seed"],
                    "viewport": result["viewport"],
                    "grid": result["grid"],
                    "settlements": result.get("settlements", []),
                })
                logger.info(
                    "Repeat %d/%d  seed=%d vp=(%d,%d %dx%d)",
                    i + 1, len(repeat_plan), q["seed"],
                    q["x"], q["y"], q["w"], q["h"],
                )
                time.sleep(self.SIMULATE_DELAY)
            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code == 429:
                    logger.warning("Rate limit hit during repeat phase")
                    break
                logger.error("Repeat HTTP %d: %s", code, e)
                break
            except Exception as e:
                logger.error("Repeat query failed: %s", e)
                break

        self._save_observations(round_id, repeat_obs)

        # ============================================================
        # PHASE 2: Deep queries — remaining budget, spread by dynamism
        # ============================================================
        try:
            budget_info = self.get_budget()
            remaining = budget_info["queries_max"] - budget_info["queries_used"]
        except Exception:
            remaining = max(0, remaining - len(repeat_obs))

        if remaining <= 0:
            # Still submit with empirical data from repeats
            empirical = self.build_empirical_distributions(repeat_obs, states, W, H)
            transitions = self.learn_transitions(repeat_obs, states)
            for si in range(n_seeds):
                mc_pred = self._run_mc_prediction(analyses[si], sim_params, n_runs=500)
                pred = self.build_prediction(si, analyses[si], repeat_obs, transitions, W, H,
                                             mc_pred=mc_pred, empirical=empirical)
                try:
                    self.submit(round_id, si, pred.tolist())
                    time.sleep(self.SUBMIT_DELAY)
                except Exception as e:
                    logger.warning("Post-repeat submit seed %d failed: %s", si, e)
            return {
                "status": "completed_repeat_only",
                "round_id": round_id,
                "round_type": self.round_type,
                "queries_executed": len(repeat_obs),
            }

        deep_plan = self.plan_deep_queries(
            analyses, repeat_obs, remaining, n_seeds, W, H
        )
        logger.info("Deep phase: %d queries planned, budget=%d", len(deep_plan), remaining)

        all_obs = list(repeat_obs)
        for i, q in enumerate(deep_plan):
            try:
                result = self.simulate(
                    round_id, q["seed"], q["x"], q["y"], q["w"], q["h"]
                )
                all_obs.append({
                    "seed": q["seed"],
                    "viewport": result["viewport"],
                    "grid": result["grid"],
                    "settlements": result.get("settlements", []),
                })
                used = result.get("queries_used", "?")
                mx = result.get("queries_max", "?")
                logger.info(
                    "Deep %d/%d  seed=%d vp=(%d,%d %dx%d)  budget=%s/%s",
                    i + 1, len(deep_plan), q["seed"],
                    q["x"], q["y"], q["w"], q["h"], used, mx,
                )
                time.sleep(self.SIMULATE_DELAY)
            except requests.exceptions.HTTPError as e:
                code = e.response.status_code if e.response is not None else 0
                if code == 429:
                    logger.warning("Rate/budget limit hit at deep query %d — stopping", i + 1)
                    break
                logger.error("HTTP %d at deep query %d: %s", code, i + 1, e)
                break
            except Exception as e:
                logger.error("Deep query %d failed: %s", i + 1, e)
                break

        logger.info("Total observations: %d (scout=%d, deep=%d)",
                     len(all_obs), len(scout_obs), len(all_obs) - len(scout_obs))

        self._save_observations(round_id, all_obs)

        # ============================================================
        # PHASE 3: Final predictions — refine SimParams + MC + blend
        # ============================================================
        transitions = self.learn_transitions(all_obs, states)
        empirical = self.build_empirical_distributions(all_obs, states, W, H)

        # Refine sim params from observations: scale spawn_prob based on
        # observed expansion vs expected
        round_est = self.estimate_round_params(transitions)
        exp_ratio = round_est.get("expansion_ratio", 1.0)
        if exp_ratio != 1.0 and 0.2 < exp_ratio < 5.0:
            refined_params = SimParams(
                spawn_prob=sim_params.spawn_prob * exp_ratio,
                spawn_pop_threshold=sim_params.spawn_pop_threshold,
                death_base_rate=sim_params.death_base_rate,
                death_food_factor=sim_params.death_food_factor,
                pop_growth_rate=sim_params.pop_growth_rate,
                food_base_regen=sim_params.food_base_regen,
                spawn_max_per_step=sim_params.spawn_max_per_step,
                port_prob=sim_params.port_prob,
                wealth_decay=sim_params.wealth_decay,
                wealth_coastal_rate=sim_params.wealth_coastal_rate,
            )
            logger.info("Refined spawn_prob: %.3f → %.3f (exp_ratio=%.2f)",
                         sim_params.spawn_prob, refined_params.spawn_prob, exp_ratio)
        else:
            refined_params = sim_params

        results = []
        for si in range(n_seeds):
            mc_pred = self._run_mc_prediction(analyses[si], refined_params, n_runs=500)
            pred = self.build_prediction(si, analyses[si], all_obs, transitions, W, H,
                                         mc_pred=mc_pred, empirical=empirical)
            try:
                resp = self.submit(round_id, si, pred.tolist())
                results.append({"seed": si, "status": resp.get("status", "ok")})
                logger.info("Submitted seed %d", si)
                time.sleep(self.SUBMIT_DELAY)
            except Exception as e:
                results.append({"seed": si, "error": str(e)})
                logger.error("Submit seed %d failed: %s", si, e)

        return {
            "status": "completed",
            "round_id": round_id,
            "round_type": self.round_type,
            "queries_executed": len(all_obs),
            "scout_queries": len(scout_obs),
            "deep_queries": len(all_obs) - len(scout_obs),
            "results": results,
        }
