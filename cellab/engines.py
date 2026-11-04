"""
Module 2: CelLab NumPy Simulation Engines (engines.py).
Implements BaseEngine and specialized vectorized subclasses:
- Elementary1DEngine
- Synchronous2DEngine
- Agent2DEngine
- Sandpile2DEngine
- Continuous2DEngine
"""

from __future__ import annotations

import abc
import copy
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from cellab.config_parser import AutomatonConfig, AutomatonConfigParser


def count_neighbors(
    grid: np.ndarray,
    target_state: int,
    neighborhood_type: str = "moore",
    boundary: str = "wrap",
    radius: int = 1,
) -> np.ndarray:
    """
    Pure NumPy neighbor counter using np.roll or np.pad slices with zero external C-library issues.
    Counts how many neighbors around each cell equal `target_state`.
    """
    mask = (grid == target_state).astype(np.int32)
    rows, cols = mask.shape
    ntype = neighborhood_type.lower()
    bmode = boundary.lower()

    offsets: List[Tuple[int, int]] = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue
            if ntype in ("von_neumann", "vn"):
                if abs(dr) + abs(dc) <= radius:
                    offsets.append((dr, dc))
            else:
                # Default Moore neighborhood
                offsets.append((dr, dc))

    total = np.zeros((rows, cols), dtype=np.int32)

    if bmode == "wrap":
        for dr, dc in offsets:
            total += np.roll(mask, shift=(-dr, -dc), axis=(0, 1))
    else:
        pad_mode = "edge" if bmode == "reflect" else "constant"
        padded = np.pad(mask, pad_width=radius, mode=pad_mode)
        for dr, dc in offsets:
            r_start = radius + dr
            c_start = radius + dc
            total += padded[r_start : r_start + rows, c_start : c_start + cols]

    return total


def shift_array_2d(arr: np.ndarray, dr: int, dc: int, boundary: str = "wrap") -> np.ndarray:
    """Shifts `arr` so that element at (r + dr, c + dc) moves to (r, c)."""
    rows, cols = arr.shape
    bmode = boundary.lower()
    if bmode == "wrap":
        return np.roll(arr, shift=(-dr, -dc), axis=(0, 1))
    pad_mode = "edge" if bmode == "reflect" else "constant"
    rmax = max(abs(dr), abs(dc), 1)
    padded = np.pad(arr, pad_width=rmax, mode=pad_mode)
    r_start = rmax + dr
    c_start = rmax + dc
    return padded[r_start : r_start + rows, c_start : c_start + cols]


class BaseEngine(abc.ABC):
    """
    Abstract base class for all CelLab NumPy simulation engines.
    Attributes:
        config: AutomatonConfig instance
        grid: 2D np.ndarray for visualization
        generation: int generation counter
        initial_grid: copy of initial grid for Reset
    """

    def __init__(self, config: AutomatonConfig, initial_grid: Optional[np.ndarray] = None):
        self.config = config
        self.generation: int = 0
        if initial_grid is not None:
            self.initial_grid = initial_grid.copy()
        else:
            self.initial_grid = AutomatonConfigParser.generate_initial_grid(config)
        self.grid: np.ndarray = self.initial_grid.copy()

    @abc.abstractmethod
    def step(self) -> np.ndarray:
        """Advances simulation by 1 generation and returns a copy of the new state."""
        pass

    def reset(self) -> np.ndarray:
        """Restores grid to initial_grid and resets generation = 0."""
        self.grid = self.initial_grid.copy()
        self.generation = 0
        return self.grid.copy()

    def set_state(
        self,
        grid_snapshot: np.ndarray,
        gen_num: int,
        agent_state: Optional[Any] = None,
    ) -> None:
        """Used by the Back button to restore a prior state."""
        self.grid = grid_snapshot.copy()
        self.generation = int(gen_num)

    def get_agent_state_copy(self) -> Any:
        """Returns a deep copy of auxiliary agent/row state if applicable."""
        return None

    def get_state_hash(self) -> bytes:
        """Returns self.grid.tobytes() (plus agent state bytes if applicable) for cycle/stability detection."""
        if self.grid.dtype in (np.float32, np.float64):
            # Round continuous grids slightly so floating-point noise doesn't prevent stability detection
            return np.round(self.grid, 6).tobytes()
        return self.grid.tobytes()

    def get_display_grid(self) -> np.ndarray:
        """Returns the 2D array to be rendered on GUI viewport."""
        return self.grid

    def clone(self) -> "BaseEngine":
        """Creates an independent copy of this engine reset to initial state for analysis."""
        cloned = create_engine(self.config, initial_grid=self.initial_grid.copy())
        return cloned


class Elementary1DEngine(BaseEngine):
    """
    1D Elementary Cellular Automaton (e.g., Rule 30).
    Maintains a 1D active row of length W and a 2D display matrix of shape (H, W)
    so the user sees the spacetime diagram grow row-by-row.
    """

    def __init__(self, config: AutomatonConfig, initial_grid: Optional[np.ndarray] = None):
        super().__init__(config, initial_grid)
        self.rows, self.cols = self.grid.shape
        self.rule_number: int = int(config.rule_number)
        self.active_row: np.ndarray = self.grid[0, :].astype(np.int32).copy()

    def reset(self) -> np.ndarray:
        super().reset()
        self.active_row = self.grid[0, :].astype(np.int32).copy()
        return self.grid.copy()

    def set_state(
        self,
        grid_snapshot: np.ndarray,
        gen_num: int,
        agent_state: Optional[Any] = None,
    ) -> None:
        super().set_state(grid_snapshot, gen_num, agent_state)
        if agent_state is not None and isinstance(agent_state, np.ndarray):
            self.active_row = agent_state.copy()
        else:
            row_idx = min(self.generation, self.rows - 1)
            self.active_row = self.grid[row_idx, :].astype(np.int32).copy()

    def get_agent_state_copy(self) -> Any:
        return self.active_row.copy()

    def step(self) -> np.ndarray:
        row = self.active_row
        if self.config.grid.boundary == "wrap":
            left = np.roll(row, 1)
            right = np.roll(row, -1)
        else:
            left = np.zeros_like(row)
            right = np.zeros_like(row)
            left[1:] = row[:-1]
            right[:-1] = row[1:]

        center = row
        pattern = (left << 2) | (center << 1) | right
        next_row = ((self.rule_number >> pattern) & 1).astype(np.int32)

        self.active_row = next_row
        self.generation += 1

        if self.generation < self.rows:
            self.grid[self.generation, :] = next_row
        else:
            self.grid[:-1, :] = self.grid[1:, :]
            self.grid[-1, :] = next_row

        return self.grid.copy()


class Synchronous2DEngine(BaseEngine):
    """
    Declarative 2D Synchronous Cellular Automaton Engine.
    Supports Life, Brian's Brain, Cyclic CA, Forest Fire, Voter, Crystal Growth.
    Evaluates declarative JSON rules using boolean masks on a double buffer.
    """

    def __init__(self, config: AutomatonConfig, initial_grid: Optional[np.ndarray] = None):
        super().__init__(config, initial_grid)
        self.rng = np.random.default_rng(config.grid.seed)

    def step(self) -> np.ndarray:
        next_grid = self.grid.copy()
        rules = self.config.rules
        if not isinstance(rules, list):
            rules = [rules] if isinstance(rules, dict) else []

        ntype = self.config.neighborhood.type
        radius = self.config.neighborhood.radius
        boundary = self.config.grid.boundary

        # Cache neighbor counts per state on demand for efficiency
        neighbor_cache: Dict[int, np.ndarray] = {}

        def get_neighbor_count(state_val: int) -> np.ndarray:
            if state_val not in neighbor_cache:
                neighbor_cache[state_val] = count_neighbors(
                    self.grid, state_val, ntype, boundary, radius
                )
            return neighbor_cache[state_val]

        for rule in rules:
            if not isinstance(rule, dict):
                continue

            cond = rule.get("condition", {"type": "always"})
            cond_type = str(cond.get("type", "always")).strip().lower()

            # Handle global/cyclic/majority rules that don't require fixed from/to
            if cond_type == "cyclic":
                num_states = int(cond.get("num_states", len(self.config.states)))
                threshold = int(cond.get("threshold", cond.get("value", 1)))
                for s in range(num_states):
                    target_s = (s + 1) % num_states
                    cnt = get_neighbor_count(target_s)
                    mask = (self.grid == s) & (cnt >= threshold)
                    next_grid[mask] = target_s
                continue

            if cond_type == "majority":
                state_ids = sorted(self.config.states.values())
                counts_stack = np.stack([get_neighbor_count(s) for s in state_ids], axis=0)
                max_idx = np.argmax(counts_stack, axis=0)
                max_val = np.max(counts_stack, axis=0)
                chosen_states = np.array(state_ids, dtype=np.int32)[max_idx]
                has_neighbors = max_val > 0
                next_grid[has_neighbors] = chosen_states[has_neighbors]
                continue

            from_state = rule.get("from")
            to_state = rule.get("to")
            if to_state is None:
                continue
            to_state = int(to_state)

            if from_state is None or from_state == "*":
                from_mask = np.ones(self.grid.shape, dtype=bool)
            elif isinstance(from_state, (list, tuple)):
                from_mask = np.isin(self.grid, [int(x) for x in from_state])
            else:
                from_mask = self.grid == int(from_state)

            if not np.any(from_mask):
                continue

            if cond_type == "always":
                cond_mask = np.ones(self.grid.shape, dtype=bool)

            elif cond_type == "neighbor_count":
                target_state = int(cond.get("state", 1))
                op = str(cond.get("op", "==")).strip()
                val = cond.get("value", 0)
                cnt = get_neighbor_count(target_state)
                cond_mask = self._eval_comparison(cnt, op, val)

            elif cond_type == "probabilistic":
                prob = float(cond.get("prob", cond.get("probability", 0.5)))
                cond_mask = self.rng.random(self.grid.shape) < prob
                if "state" in cond and "op" in cond and "value" in cond:
                    target_state = int(cond["state"])
                    cnt = get_neighbor_count(target_state)
                    cond_mask = cond_mask & self._eval_comparison(cnt, str(cond["op"]), cond["value"])

            else:
                cond_mask = np.zeros(self.grid.shape, dtype=bool)

            next_grid[from_mask & cond_mask] = to_state

        self.grid = next_grid
        self.generation += 1
        return self.grid.copy()

    @staticmethod
    def _eval_comparison(cnt: np.ndarray, op: str, val: Any) -> np.ndarray:
        if op == "==":
            return cnt == int(val)
        elif op == "!=":
            return cnt != int(val)
        elif op == ">=":
            return cnt >= int(val)
        elif op == "<=":
            return cnt <= int(val)
        elif op == ">":
            return cnt > int(val)
        elif op == "<":
            return cnt < int(val)
        elif op == "in":
            if isinstance(val, (list, tuple, set)):
                return np.isin(cnt, [int(v) for v in val])
            return cnt == int(val)
        raise ValueError(f"Unsupported comparison operator in neighbor_count rule: '{op}'")


class Agent2DEngine(BaseEngine):
    """
    2D Agent-based Cellular Automaton Engine (e.g., Langton's Ant).
    Tracks self.agents (list of dicts with row, col, dir in [0, 1, 2, 3] for N, E, S, W).
    In step(), updates the cell under each agent, rotates direction (+1 for Right, -1 for Left mod 4),
    and steps forward with boundary wrapping/clamping.
    """

    # 0: North (-1, 0), 1: East (0, +1), 2: South (+1, 0), 3: West (0, -1)
    DELTAS = {
        0: (-1, 0),
        1: (0, 1),
        2: (1, 0),
        3: (0, -1),
    }

    def __init__(self, config: AutomatonConfig, initial_grid: Optional[np.ndarray] = None):
        super().__init__(config, initial_grid)
        self.rows, self.cols = self.grid.shape
        self.initial_agents = self._parse_initial_agents(config)
        self.agents: List[Dict[str, int]] = copy.deepcopy(self.initial_agents)
        self.transition_map = self._parse_agent_rules(config)

    def _parse_initial_agents(self, config: AutomatonConfig) -> List[Dict[str, int]]:
        agents = []
        if config.agents:
            for ag in config.agents:
                agents.append(
                    {
                        "row": int(ag.get("row", self.rows // 2)),
                        "col": int(ag.get("col", self.cols // 2)),
                        "dir": int(ag.get("dir", 0)) % 4,
                    }
                )
        else:
            agents.append({"row": self.rows // 2, "col": self.cols // 2, "dir": 0})
        return agents

    def _parse_agent_rules(self, config: AutomatonConfig) -> Dict[int, Tuple[int, int]]:
        """
        Maps current cell state -> (next_cell_state, delta_dir).
        delta_dir: +1 for R (Right), -1 for L (Left), 0 for N (None), +2 for U (U-turn).
        """
        tmap: Dict[int, Tuple[int, int]] = {}
        rules = config.rules

        def turn_to_delta(turn_val: Union[str, int]) -> int:
            if isinstance(turn_val, int):
                return turn_val
            t = str(turn_val).strip().upper()
            if t in ("R", "RIGHT", "+1", "1"):
                return 1
            if t in ("L", "LEFT", "-1"):
                return -1
            if t in ("U", "UTURN", "BACK", "2"):
                return 2
            return 0

        if isinstance(rules, list) and len(rules) > 0:
            for r in rules:
                if isinstance(r, dict) and "from" in r and "to" in r:
                    f_state = int(r["from"])
                    t_state = int(r["to"])
                    turn = r.get("turn", r.get("condition", {}).get("turn", "R" if f_state == 0 else "L"))
                    tmap[f_state] = (t_state, turn_to_delta(turn))
        elif isinstance(rules, dict):
            if "turn_string" in rules or "rules" in rules:
                rstr = str(rules.get("turn_string", rules.get("rules", "RL"))).upper()
                n = len(rstr)
                for idx, ch in enumerate(rstr):
                    tmap[idx] = ((idx + 1) % n, turn_to_delta(ch))
            else:
                for k, v in rules.items():
                    if isinstance(v, dict):
                        f_state = int(k)
                        t_state = int(v.get("to", v.get("next_state", (f_state + 1) % 2)))
                        turn = v.get("turn", "R")
                        tmap[f_state] = (t_state, turn_to_delta(turn))

        if not tmap:
            # Default Langton's Ant: state 0 -> 1 (Right), state 1 -> 0 (Left)
            tmap = {0: (1, 1), 1: (0, -1)}

        return tmap

    def reset(self) -> np.ndarray:
        super().reset()
        self.agents = copy.deepcopy(self.initial_agents)
        return self.grid.copy()

    def set_state(
        self,
        grid_snapshot: np.ndarray,
        gen_num: int,
        agent_state: Optional[Any] = None,
    ) -> None:
        super().set_state(grid_snapshot, gen_num, agent_state)
        if agent_state is not None:
            self.agents = copy.deepcopy(agent_state)

    def get_agent_state_copy(self) -> Any:
        return copy.deepcopy(self.agents)

    def get_state_hash(self) -> bytes:
        agent_tuple = tuple((a["row"], a["col"], a["dir"]) for a in self.agents)
        return self.grid.tobytes() + str(agent_tuple).encode("ascii")

    def step(self) -> np.ndarray:
        bmode = self.config.grid.boundary
        num_defined = max(2, len(self.transition_map))

        for agent in self.agents:
            r, c, d = agent["row"], agent["col"], agent["dir"]
            cell_state = int(self.grid[r, c])
            next_state, delta_dir = self.transition_map.get(
                cell_state, ((cell_state + 1) % num_defined, 1)
            )

            # Rotate direction (+1 for Right, -1 for Left mod 4)
            new_dir = (d + delta_dir) % 4
            # Update cell under agent
            self.grid[r, c] = next_state

            # Step forward
            dr, dc = self.DELTAS[new_dir]
            nr, nc = r + dr, c + dc
            if bmode == "wrap":
                nr %= self.rows
                nc %= self.cols
            else:
                nr = max(0, min(self.rows - 1, nr))
                nc = max(0, min(self.cols - 1, nc))

            agent["row"] = nr
            agent["col"] = nc
            agent["dir"] = new_dir

        self.generation += 1
        return self.grid.copy()


class Sandpile2DEngine(BaseEngine):
    """
    Abelian Sandpile 2D Engine (Bak-Tang-Wiesenfeld).
    Vectorized toppling where unstable = self.grid >= 4;
    subtracts 4 from unstable cells and adds 1 to N, S, E, W neighbors using np.roll.
    """

    def __init__(self, config: AutomatonConfig, initial_grid: Optional[np.ndarray] = None):
        super().__init__(config, initial_grid)
        self.threshold = 4

    def step(self) -> np.ndarray:
        unstable = (self.grid >= self.threshold).astype(np.int32)
        if np.any(unstable):
            self.grid = self.grid - 4 * unstable

            north = np.roll(unstable, shift=-1, axis=0)
            south = np.roll(unstable, shift=1, axis=0)
            west = np.roll(unstable, shift=-1, axis=1)
            east = np.roll(unstable, shift=1, axis=1)

            if self.config.grid.boundary != "wrap":
                # Open/absorbing boundary: grains toppling off the edges disappear
                north[-1, :] = 0
                south[0, :] = 0
                west[:, -1] = 0
                east[:, 0] = 0

            self.grid = self.grid + north + south + east + west

        self.generation += 1
        return self.grid.copy()


class Continuous2DEngine(BaseEngine):
    """
    Continuous 2D Engine (e.g., Heat Diffusion).
    Updates grid via: self.grid = 0.2 * (self.grid + north + south + east + west).
    """

    def __init__(self, config: AutomatonConfig, initial_grid: Optional[np.ndarray] = None):
        super().__init__(config, initial_grid)
        self.grid = self.grid.astype(np.float64)
        self.initial_grid = self.initial_grid.astype(np.float64)

    def step(self) -> np.ndarray:
        bmode = self.config.grid.boundary
        if bmode == "wrap":
            north = np.roll(self.grid, shift=-1, axis=0)
            south = np.roll(self.grid, shift=1, axis=0)
            west = np.roll(self.grid, shift=-1, axis=1)
            east = np.roll(self.grid, shift=1, axis=1)
        else:
            north = shift_array_2d(self.grid, -1, 0, bmode)
            south = shift_array_2d(self.grid, 1, 0, bmode)
            west = shift_array_2d(self.grid, 0, -1, bmode)
            east = shift_array_2d(self.grid, 0, 1, bmode)

        self.grid = 0.2 * (self.grid + north + south + east + west)
        self.generation += 1
        return self.grid.copy()


def create_engine(config: AutomatonConfig, initial_grid: Optional[np.ndarray] = None) -> BaseEngine:
    """Factory function instantiating the appropriate BaseEngine subclass."""
    etype = config.engine_type.lower()
    if etype == "elementary_1d":
        return Elementary1DEngine(config, initial_grid)
    elif etype == "synchronous_2d":
        return Synchronous2DEngine(config, initial_grid)
    elif etype == "agent_2d":
        return Agent2DEngine(config, initial_grid)
    elif etype == "sandpile_2d":
        return Sandpile2DEngine(config, initial_grid)
    elif etype == "continuous_2d":
        return Continuous2DEngine(config, initial_grid)
    else:
        raise ValueError(f"Unsupported engine_type: '{etype}'")


# Alias for backward compatibility
BaseAutomaton = BaseEngine
