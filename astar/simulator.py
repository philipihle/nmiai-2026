
"""
Astar Island Simulator
======================
Approximates the Norse civilization simulation based on observed replay behavior.
Used to generate training data via Monte Carlo and calibrate to round-specific parameters.

Optimized with numpy arrays for fast batch simulation.
"""

import multiprocessing as mp
import numpy as np
from typing import Optional

TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# Module-level globals for multiprocessing workers
_worker_grid = None
_worker_settlements = None
_worker_params = None
_worker_gt = None


def _init_worker(grid, settlements, params, gt=None):
    global _worker_grid, _worker_settlements, _worker_params, _worker_gt
    _worker_grid = grid
    _worker_settlements = settlements
    _worker_params = params
    _worker_gt = gt


def _grid_to_counts(grid):
    """Vectorized terrain counting: grid (H,W) → counts (H,W,6)."""
    H, W = grid.shape
    counts = np.zeros((H, W, 6), dtype=np.float32)
    # Map terrain values to class indices using a lookup table
    lookup = np.zeros(20, dtype=np.int32)  # max terrain value
    for tv, cls in TERRAIN_TO_CLASS.items():
        if tv < 20:
            lookup[tv] = cls
    flat_cls = lookup[np.clip(grid, 0, 19)]
    for c in range(6):
        counts[:, :, c] = (flat_cls == c).astype(np.float32)
    return counts


def _run_single_mc(seed):
    """Run one MC simulation. Used by Pool workers."""
    sim = AstarSimulator(_worker_grid, _worker_settlements,
                         params=_worker_params, seed=seed)
    final = sim.run(50)
    return _grid_to_counts(final)


def _eval_params_worker(args):
    """Evaluate one param combo by settlement count. Used by parallel grid search."""
    sp, th, db, df = args
    params = SimParams(
        spawn_prob=sp, spawn_pop_threshold=th,
        death_base_rate=db, death_food_factor=df,
        spawn_max_per_step=40, port_prob=0.10,
        wealth_decay=0.0, wealth_coastal_rate=0.01,
    )
    sim = AstarSimulator(_worker_grid, _worker_settlements, params=params, seed=42)
    final = sim.run(50)
    ns = int(np.sum((final == 1) | (final == 2)))
    nr = int(np.sum(final == 3))
    np_ = int(np.sum(final == 2))
    return (args, ns, nr, np_)


def _eval_combo_with_mc(combo):
    """Evaluate one param combo with MC scoring. Each worker runs MC sequentially."""
    from backtest import score_prediction
    sp, th, db, df, pgr, fbr = combo
    params = SimParams(
        spawn_prob=sp, spawn_pop_threshold=th,
        death_base_rate=db, death_food_factor=df,
        pop_growth_rate=pgr, food_base_regen=fbr,
        spawn_max_per_step=40, port_prob=0.10,
        wealth_decay=0.0, wealth_coastal_rate=0.01,
    )
    sim = AstarSimulator(_worker_grid, _worker_settlements, params=params, seed=123)
    probs = sim.monte_carlo(n_runs=20, n_steps=50, n_workers=1)
    score = score_prediction(probs, _worker_gt)
    return (combo, score)


class SimParams:
    """Hidden parameters that vary per round."""

    def __init__(self, **kwargs):
        self.pop_growth_rate = kwargs.get("pop_growth_rate", 0.10)
        self.pop_carrying_cap = kwargs.get("pop_carrying_cap", 3.0)
        self.pop_noise = kwargs.get("pop_noise", 0.02)
        self.food_base_regen = kwargs.get("food_base_regen", 0.15)
        self.food_pop_drain = kwargs.get("food_pop_drain", 0.08)
        self.food_competition = kwargs.get("food_competition", 0.03)
        self.food_forest_bonus = kwargs.get("food_forest_bonus", 0.02)
        self.food_noise = kwargs.get("food_noise", 0.05)
        self.def_growth_rate = kwargs.get("def_growth_rate", 0.055)
        self.def_max = kwargs.get("def_max", 1.0)
        self.wealth_coastal_rate = kwargs.get("wealth_coastal_rate", 0.005)
        self.wealth_port_rate = kwargs.get("wealth_port_rate", 0.02)
        self.wealth_decay = kwargs.get("wealth_decay", 0.01)
        self.death_base_rate = kwargs.get("death_base_rate", 0.03)
        self.death_food_factor = kwargs.get("death_food_factor", 0.15)
        self.death_defense_factor = kwargs.get("death_defense_factor", 0.05)
        self.death_competition_factor = kwargs.get("death_competition_factor", 0.02)
        self.spawn_pop_threshold = kwargs.get("spawn_pop_threshold", 0.8)
        self.spawn_prob = kwargs.get("spawn_prob", 0.15)
        self.spawn_max_per_step = kwargs.get("spawn_max_per_step", 20)
        self.spawn_init_pop = kwargs.get("spawn_init_pop", 0.5)
        self.spawn_init_def = kwargs.get("spawn_init_def", 0.2)
        self.ruin_to_settle_prob = kwargs.get("ruin_to_settle_prob", 0.47)
        self.ruin_to_empty_prob = kwargs.get("ruin_to_empty_prob", 0.35)
        self.port_wealth_threshold = kwargs.get("port_wealth_threshold", 0.0)
        self.port_prob = kwargs.get("port_prob", 0.05)
        # Raiding (default OFF — grid search decides)
        self.raid_base_prob = kwargs.get("raid_base_prob", 0.0)
        self.raid_desperate_factor = kwargs.get("raid_desperate_factor", 0.15)
        self.raid_range = kwargs.get("raid_range", 3)
        self.longship_range_bonus = kwargs.get("longship_range_bonus", 3)
        self.raid_loot_factor = kwargs.get("raid_loot_factor", 0.3)
        self.raid_damage_factor = kwargs.get("raid_damage_factor", 0.3)
        self.raid_conquest_prob = kwargs.get("raid_conquest_prob", 0.05)
        # Port survival bonus (replay: ports die at 3% vs non-ports 6%)
        self.port_survival_bonus = kwargs.get("port_survival_bonus", 0.0)
        # Death wave (default OFF — captures boom-bust without modeling exact cascade)
        self.death_wave_amplitude = kwargs.get("death_wave_amplitude", 0.0)
        self.death_wave_period = kwargs.get("death_wave_period", 3)
        # Weighted spawn selection (default = neutral/uniform)
        self.spawn_neighbor_weight = kwargs.get("spawn_neighbor_weight", 0.0)
        self.spawn_ruin_preference = kwargs.get("spawn_ruin_preference", 17.68)
        self.spawn_forest_preference = kwargs.get("spawn_forest_preference", 1.0)
        self.spawn_forest_adj_weight = kwargs.get("spawn_forest_adj_weight", 0.0)
        self.spawn_coastal_preference = kwargs.get("spawn_coastal_preference", 1.0)
        # Deterministic strategies (all default OFF for backward compat)
        self.deterministic_death = kwargs.get("deterministic_death", False)
        self.death_threshold = kwargs.get("death_threshold", 0.5)
        self.deterministic_spawn_location = kwargs.get("deterministic_spawn_location", False)
        self.deterministic_parent_selection = kwargs.get("deterministic_parent_selection", False)
        # Food model fixes (defaults = no change for backward compat)
        self.food_floor = kwargs.get("food_floor", 0.0)
        self.food_share_rate = kwargs.get("food_share_rate", 0.0)


class AstarSimulator:
    """Fast numpy-based simulator."""

    def __init__(self, grid: np.ndarray, settlements: list,
                 params: Optional[SimParams] = None, seed: Optional[int] = None):
        self.H, self.W = grid.shape
        self.initial_grid = grid.copy()
        self.initial_settlements = [s.copy() for s in settlements]
        self.params = params or SimParams()
        self.base_seed = seed or 0

        # Static masks
        self.ocean_mask = grid == 10
        self.mountain_mask = grid == 5
        land = ~self.ocean_mask
        self.coastal = np.zeros((self.H, self.W), dtype=bool)
        if self.H > 1:
            self.coastal[1:] |= land[1:] & self.ocean_mask[:-1]
            self.coastal[:-1] |= land[:-1] & self.ocean_mask[1:]
        if self.W > 1:
            self.coastal[:, 1:] |= land[:, 1:] & self.ocean_mask[:, :-1]
            self.coastal[:, :-1] |= land[:, :-1] & self.ocean_mask[:, 1:]

        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.RandomState(seed if seed is not None else self.base_seed)
        self._step_count = 0
        self.grid = self.initial_grid.copy()
        # Settlement arrays: parallel arrays for vectorized ops
        self.s_x = []
        self.s_y = []
        self.s_pop = []
        self.s_food = []
        self.s_wealth = []
        self.s_def = []
        self.s_port = []
        self.s_owner = []
        for s in self.initial_settlements:
            if not s.get("alive", True):
                continue
            self.s_x.append(s["x"])
            self.s_y.append(s["y"])
            self.s_pop.append(s.get("population", 0.5))
            self.s_food.append(s.get("food", 0.5))
            self.s_wealth.append(s.get("wealth", 0.0))
            self.s_def.append(s.get("defense", 0.2))
            self.s_port.append(s.get("has_port", False))
            self.s_owner.append(s.get("owner_id", 0))
        self._to_arrays()

    def _to_arrays(self):
        self.s_x = np.array(self.s_x, dtype=np.int32)
        self.s_y = np.array(self.s_y, dtype=np.int32)
        self.s_pop = np.array(self.s_pop, dtype=np.float64)
        self.s_food = np.array(self.s_food, dtype=np.float64)
        self.s_wealth = np.array(self.s_wealth, dtype=np.float64)
        self.s_def = np.array(self.s_def, dtype=np.float64)
        self.s_port = np.array(self.s_port, dtype=bool)
        self.s_owner = np.array(self.s_owner, dtype=np.int32)

    def _settlement_density_map(self, radius=3):
        """Build density map using scipy convolution with Manhattan diamond kernel."""
        from scipy.ndimage import convolve
        settle_grid = np.zeros((self.H, self.W), dtype=np.int32)
        for i in range(len(self.s_x)):
            settle_grid[self.s_y[i], self.s_x[i]] += 1
        # Build Manhattan diamond kernel
        size = 2 * radius + 1
        kernel = np.zeros((size, size), dtype=np.int32)
        for dy in range(size):
            for dx in range(size):
                if abs(dy - radius) + abs(dx - radius) <= radius:
                    kernel[dy, dx] = 1
        return convolve(settle_grid, kernel, mode='constant', cval=0)

    def _adj_forest_map(self):
        """Count adjacent forest cells for each cell."""
        f = (self.grid == 4).astype(np.int32)
        adj = np.zeros((self.H, self.W), dtype=np.int32)
        if self.H > 1:
            adj[1:] += f[:-1]
            adj[:-1] += f[1:]
        if self.W > 1:
            adj[:, 1:] += f[:, :-1]
            adj[:, :-1] += f[:, 1:]
        return adj

    def step(self):
        p = self.params

        # 1. Recover PREVIOUS step's ruins first (ruins persist exactly 1 step)
        self._recover_ruins()

        n = len(self.s_x)
        if n == 0:
            return

        # Density map for competition
        density = self._settlement_density_map(3)
        adj_forest = self._adj_forest_map()

        # Per-settlement values from maps
        n_nearby = density[self.s_y, self.s_x] - 1
        s_adj_forest = adj_forest[self.s_y, self.s_x]

        # 2. Update attributes (vectorized)
        growth = p.pop_growth_rate * self.s_pop * (1 - self.s_pop / p.pop_carrying_cap)
        growth += self.rng.normal(0, p.pop_noise, n)
        self.s_pop = np.maximum(0.01, self.s_pop + growth)

        food_regen = p.food_base_regen + s_adj_forest * p.food_forest_bonus
        food_drain = self.s_pop * p.food_pop_drain + n_nearby * p.food_competition
        food_delta = food_regen - food_drain + self.rng.normal(0, p.food_noise, n)
        self.s_food = np.clip(self.s_food + food_delta, 0.0, 1.0)
        # Food floor: prevent food from collapsing below a minimum
        if p.food_floor > 0:
            self.s_food = np.maximum(self.s_food, p.food_floor)

        def_growth = p.def_growth_rate * (1 - self.s_def / p.def_max)
        self.s_def = np.minimum(p.def_max, self.s_def + def_growth)

        is_coastal = self.coastal[self.s_y, self.s_x]
        self.s_wealth += is_coastal * p.wealth_coastal_rate + self.s_port * p.wealth_port_rate
        self.s_wealth = np.maximum(0, self.s_wealth - p.wealth_decay)

        # 2b. Raiding (opt-in, between attribute updates and deaths)
        if p.raid_base_prob > 0:
            self._raid_phase()

        # 3. Deaths — settlements become ruins (persist until next step)
        # Global death wave (captures boom-bust without modeling exact cascade)
        if p.death_wave_amplitude > 0:
            wave = 1.0 + p.death_wave_amplitude * np.sin(
                2 * np.pi * self._step_count / p.death_wave_period)
            wave = max(0.5, wave)
        else:
            wave = 1.0

        death_prob = (p.death_base_rate * wave
                      + p.death_food_factor * np.maximum(0, 1.0 - self.s_food)
                      - p.death_defense_factor * self.s_def
                      + p.death_competition_factor * n_nearby
                      - p.port_survival_bonus * self.s_port)
        death_prob = np.clip(death_prob, 0.01, 0.5)
        if p.deterministic_death:
            dies = death_prob > p.death_threshold
        else:
            dies = self.rng.random(n) < death_prob

        for i in np.where(dies)[0]:
            self.grid[self.s_y[i], self.s_x[i]] = 3

        alive = ~dies
        self.s_x = self.s_x[alive]
        self.s_y = self.s_y[alive]
        self.s_pop = self.s_pop[alive]
        self.s_food = self.s_food[alive]
        self.s_wealth = self.s_wealth[alive]
        self.s_def = self.s_def[alive]
        self.s_port = self.s_port[alive]
        self.s_owner = self.s_owner[alive]

        # 4. Spawn new settlements
        self._spawn_settlements()

        # 5. Ports
        self._check_ports()

        self._step_count += 1

    def _raid_phase(self):
        """Raiding: desperate settlements attack nearby enemies."""
        p = self.params
        n = len(self.s_x)
        if n < 2:
            return

        # Raid probability: base + desperate_factor * (1 - food)
        desperation = np.maximum(0, 1.0 - self.s_food)
        raid_prob = p.raid_base_prob + p.raid_desperate_factor * desperation
        wants_to_raid = self.rng.random(n) < raid_prob
        raiders = np.where(wants_to_raid)[0]
        if len(raiders) == 0:
            return

        self.rng.shuffle(raiders)

        # Precompute raiding range per settlement (ports get longship bonus)
        raid_range = np.full(n, p.raid_range, dtype=np.int32)
        raid_range[self.s_port] += p.longship_range_bonus

        for ri in raiders:
            if n < 2:
                break
            rx, ry = self.s_x[ri], self.s_y[ri]
            owner = self.s_owner[ri]
            rng = raid_range[ri]

            # Find targets: different owner, within range
            dx = np.abs(self.s_x - rx)
            dy = np.abs(self.s_y - ry)
            dist = dx + dy
            enemy = self.s_owner != owner
            in_range = (dist > 0) & (dist <= rng) & enemy

            targets = np.where(in_range)[0]
            if len(targets) == 0:
                continue

            # Pick weakest target (lowest defense)
            ti = targets[np.argmin(self.s_def[targets])]

            # Combat: attacker strength vs defender strength
            atk = self.s_pop[ri] * (1 + self.s_def[ri])
            dfn = self.s_pop[ti] * (1 + self.s_def[ti])
            win_prob = atk / (atk + dfn + 1e-10)

            if self.rng.random() < win_prob:
                # Successful raid: loot and damage
                loot_food = self.s_food[ti] * p.raid_loot_factor
                loot_wealth = self.s_wealth[ti] * p.raid_loot_factor
                self.s_food[ri] += loot_food
                self.s_wealth[ri] += loot_wealth
                self.s_food[ti] -= loot_food
                self.s_wealth[ti] -= loot_wealth
                self.s_def[ti] *= (1 - p.raid_damage_factor)

                # Conquest chance
                if self.rng.random() < p.raid_conquest_prob:
                    self.s_owner[ti] = owner

    def _recover_ruins(self):
        p = self.params
        ruin_ys, ruin_xs = np.where(self.grid == 3)
        if len(ruin_ys) == 0:
            return
        # Filter out positions with existing settlements
        settle_set = set(zip(self.s_x.tolist(), self.s_y.tolist()))
        valid = [(rx, ry) for rx, ry in zip(ruin_xs, ruin_ys) if (rx, ry) not in settle_set]
        if not valid:
            return

        rolls = self.rng.random(len(valid))
        new_x, new_y, new_pop, new_food, new_def, new_owner = [], [], [], [], [], []

        for idx, (rx, ry) in enumerate(valid):
            if rolls[idx] < p.ruin_to_settle_prob:
                self.grid[ry, rx] = 1
                new_x.append(rx)
                new_y.append(ry)
                new_pop.append(0.4)
                new_food.append(max(0.01, 0.2 + self.rng.normal(0, 0.1)))
                new_def.append(0.15)
                # Find nearest owner
                if len(self.s_x) > 0:
                    dists = np.abs(self.s_x - rx) + np.abs(self.s_y - ry)
                    new_owner.append(self.s_owner[dists.argmin()])
                else:
                    new_owner.append(0)
            elif rolls[idx] < p.ruin_to_settle_prob + p.ruin_to_empty_prob:
                self.grid[ry, rx] = 11
            else:
                self.grid[ry, rx] = 4

        if new_x:
            self.s_x = np.concatenate([self.s_x, np.array(new_x, dtype=np.int32)])
            self.s_y = np.concatenate([self.s_y, np.array(new_y, dtype=np.int32)])
            self.s_pop = np.concatenate([self.s_pop, np.array(new_pop)])
            self.s_food = np.concatenate([self.s_food, np.array(new_food)])
            self.s_wealth = np.concatenate([self.s_wealth, np.zeros(len(new_x))])
            self.s_def = np.concatenate([self.s_def, np.array(new_def)])
            self.s_port = np.concatenate([self.s_port, np.zeros(len(new_x), dtype=bool)])
            self.s_owner = np.concatenate([self.s_owner, np.array(new_owner, dtype=np.int32)])

    def _count_settlement_neighbors(self, x, y):
        """Count grid cells with value 1 or 2 within Manhattan distance 2."""
        count = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if abs(dx) + abs(dy) > 2 or (dx == 0 and dy == 0):
                    continue
                nx, ny_ = x + dx, y + dy
                if 0 <= nx < self.W and 0 <= ny_ < self.H:
                    v = self.grid[ny_, nx]
                    if v == 1 or v == 2:
                        count += 1
        return count

    def _count_adjacent_forest(self, x, y):
        """Count grid cells with value 4 among 4 cardinal neighbors."""
        count = 0
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny_ = x + dx, y + dy
            if 0 <= nx < self.W and 0 <= ny_ < self.H:
                if self.grid[ny_, nx] == 4:
                    count += 1
        return count

    def _is_coastal(self, x, y):
        """Return True if any cardinal neighbor has grid value 10."""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny_ = x + dx, y + dy
            if 0 <= nx < self.W and 0 <= ny_ < self.H:
                if self.grid[ny_, nx] == 10:
                    return True
        return False

    def _spawn_weights(self, candidates, parent_x, parent_y):
        """Compute selection weights for spawn candidates."""
        p = self.params
        weights = []
        for cx, cy in candidates:
            w = 1.0
            # Settlement neighbor count
            n_neighbors = self._count_settlement_neighbors(cx, cy)
            w *= (1.0 + p.spawn_neighbor_weight * n_neighbors)
            # Terrain preference
            terrain = self.grid[cy, cx]
            if terrain == 3:
                w *= p.spawn_ruin_preference
            elif terrain == 4:
                w *= p.spawn_forest_preference
            # Forest adjacency
            n_forest = self._count_adjacent_forest(cx, cy)
            w *= (1.0 + p.spawn_forest_adj_weight * n_forest)
            # Coastal
            if self._is_coastal(cx, cy):
                w *= p.spawn_coastal_preference
            weights.append(max(w, 0.01))
        return weights

    def _spawn_settlements(self):
        p = self.params
        n = len(self.s_x)
        if n == 0:
            return

        # Which settlements can spawn?
        if p.deterministic_parent_selection:
            can_spawn = self.s_pop >= p.spawn_pop_threshold
            spawners = np.where(can_spawn)[0]
            if len(spawners) > int(p.spawn_max_per_step):
                order = np.argsort(-self.s_pop[spawners])
                spawners = spawners[order[:int(p.spawn_max_per_step)]]
        else:
            can_spawn = (self.s_pop >= p.spawn_pop_threshold) & (self.rng.random(n) < p.spawn_prob)
            spawners = np.where(can_spawn)[0]
            self.rng.shuffle(spawners)
            spawners = spawners[:p.spawn_max_per_step]

        settle_set = set(zip(self.s_x.tolist(), self.s_y.tolist()))
        new_x, new_y, new_pop, new_food, new_def, new_owner = [], [], [], [], [], []

        # d≤3 spawn offsets: replay shows d=1: 42.6%, d=2: 38.7%, d=3: 13.4%
        _SPAWN_OFFSETS_D12 = [
            (0, 1), (0, -1), (1, 0), (-1, 0),                          # d=1
            (0, 2), (0, -2), (2, 0), (-2, 0),                          # d=2 cardinal
            (1, 1), (1, -1), (-1, 1), (-1, -1),                        # d=2 diagonal
        ]
        _SPAWN_OFFSETS_D3 = [
            (dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)
            if abs(dx) + abs(dy) == 3
        ]

        for si in spawners:
            sx, sy = int(self.s_x[si]), int(self.s_y[si])

            # Try d=1-2 first (81.3% of spawns), then d=3 (13.4%)
            candidates = []
            for dx, dy in _SPAWN_OFFSETS_D12:
                nx, ny = sx + dx, sy + dy
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    tv = self.grid[ny, nx]
                    if tv in (11, 4, 3) and (nx, ny) not in settle_set:
                        candidates.append((nx, ny))

            # d=3 candidates with lower probability (~16% of d1+d2 rate)
            if self.rng.random() < 0.16 or not candidates:
                for dx, dy in _SPAWN_OFFSETS_D3:
                    nx, ny = sx + dx, sy + dy
                    if 0 <= nx < self.W and 0 <= ny < self.H:
                        tv = self.grid[ny, nx]
                        if tv in (11, 4, 3) and (nx, ny) not in settle_set:
                            candidates.append((nx, ny))

            if not candidates:
                continue

            # Multi-spawn: allow up to 2 spawns per parent (replay shows 130 multi-spawn events)
            n_spawns = 1
            if len(candidates) >= 2 and self.rng.random() < 0.15:
                n_spawns = 2

            for _ in range(min(n_spawns, len(candidates))):
                if p.deterministic_spawn_location:
                    # Deterministic: ruins first, then closest, then by coords
                    def sort_key(c):
                        _cx, _cy = c
                        is_ruin = 1 if self.grid[_cy, _cx] == 3 else 0
                        dist = abs(_cx - sx) + abs(_cy - sy)
                        return (-is_ruin, dist, _cx, _cy)
                    candidates_sorted = sorted(range(len(candidates)), key=lambda i: sort_key(candidates[i]))
                    idx = candidates_sorted[0]
                else:
                    weights = self._spawn_weights(candidates, sx, sy)
                    total = sum(weights)
                    probs = [w / total for w in weights]
                    cum = 0.0
                    r = self.rng.random()
                    idx = len(candidates) - 1
                    for i, prob in enumerate(probs):
                        cum += prob
                        if r < cum:
                            idx = i
                            break
                cx, cy = candidates.pop(idx)
                settle_set.add((cx, cy))
                self.grid[cy, cx] = 1
                new_x.append(cx)
                new_y.append(cy)
                new_pop.append(p.spawn_init_pop)
                new_food.append(max(0.01, 0.2 + self.rng.normal(0, 0.1)))
                new_def.append(p.spawn_init_def)
                new_owner.append(int(self.s_owner[si]))

        if new_x:
            self.s_x = np.concatenate([self.s_x, np.array(new_x, dtype=np.int32)])
            self.s_y = np.concatenate([self.s_y, np.array(new_y, dtype=np.int32)])
            self.s_pop = np.concatenate([self.s_pop, np.array(new_pop)])
            self.s_food = np.concatenate([self.s_food, np.array(new_food)])
            self.s_wealth = np.concatenate([self.s_wealth, np.zeros(len(new_x))])
            self.s_def = np.concatenate([self.s_def, np.array(new_def)])
            self.s_port = np.concatenate([self.s_port, np.zeros(len(new_x), dtype=bool)])
            self.s_owner = np.concatenate([self.s_owner, np.array(new_owner, dtype=np.int32)])

    def _check_ports(self):
        p = self.params
        n = len(self.s_x)
        if n == 0:
            return
        is_coastal = self.coastal[self.s_y, self.s_x]
        can_port = (~self.s_port) & is_coastal & (self.s_wealth >= p.port_wealth_threshold)
        becomes_port = can_port & (self.rng.random(n) < p.port_prob)
        self.s_port |= becomes_port
        for i in np.where(becomes_port)[0]:
            self.grid[self.s_y[i], self.s_x[i]] = 2

    def run(self, n_steps: int = 50) -> np.ndarray:
        for _ in range(n_steps):
            self.step()
        return self.grid.copy()

    def monte_carlo(self, n_runs: int = 200, n_steps: int = 50,
                    n_workers: int = 0) -> np.ndarray:
        """Run Monte Carlo. n_workers=0 means use all CPUs."""
        seeds = [self.base_seed + i * 7919 for i in range(n_runs)]

        if n_workers is None or n_workers == 1:
            # Sequential fallback
            counts = np.zeros((self.H, self.W, 6), dtype=np.float32)
            for seed in seeds:
                self.reset(seed=seed)
                final = self.run(n_steps)
                counts += _grid_to_counts(final)
        else:
            # Parallel
            n_cpus = n_workers if n_workers > 0 else mp.cpu_count()
            with mp.Pool(n_cpus, initializer=_init_worker,
                         initargs=(self.initial_grid, self.initial_settlements,
                                   self.params)) as pool:
                results = pool.map(_run_single_mc, seeds)
            counts = np.sum(results, axis=0)

        probs = counts / n_runs
        probs = np.maximum(probs, 0.001)
        probs /= probs.sum(axis=-1, keepdims=True)
        return probs

    @staticmethod
    def parallel_grid_search(grid, settlements, param_combos, target_s=262,
                             target_r=12, target_p=18, n_workers=0):
        """Parallel grid search over parameter combinations (single-run, fast).

        param_combos: list of (spawn_prob, spawn_threshold, death_base, death_food)
        Returns sorted list of (diff, params, n_settle, n_ruin, n_port).
        """
        n_cpus = n_workers if n_workers > 0 else mp.cpu_count()
        with mp.Pool(n_cpus, initializer=_init_worker,
                     initargs=(grid, settlements, None, None)) as pool:
            results = pool.map(_eval_params_worker, param_combos)

        scored = []
        for (args, ns, nr, np_) in results:
            diff = abs(ns - target_s) + abs(nr - target_r) * 3 + abs(np_ - target_p) * 2
            scored.append((diff, args, ns, nr, np_))
        scored.sort()
        return scored

    @staticmethod
    def parallel_mc_grid_search(grid, settlements, gt, param_combos, n_workers=0):
        """Parallel grid search with MC scoring.

        Each worker evaluates one param combo with 20 MC runs.
        60 combos run simultaneously on 60 CPUs.

        param_combos: list of (spawn_prob, threshold, death_base, death_food, pop_growth, food_regen)
        gt: ground truth (H, W, 6) array
        Returns sorted list of (score, combo).
        """
        n_cpus = n_workers if n_workers > 0 else mp.cpu_count()
        with mp.Pool(n_cpus, initializer=_init_worker,
                     initargs=(grid, settlements, None, gt)) as pool:
            results = pool.map(_eval_combo_with_mc, param_combos)

        scored = [(score, combo) for (combo, score) in results]
        scored.sort(reverse=True)  # highest score first
        return scored
