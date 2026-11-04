"""
Module 1: CelLab Configuration Parser and Initial Grid Generator (config_parser.py).
Implements AutomatonConfigParser and AutomatonConfig following the declarative JSON schema.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


VALID_ENGINE_TYPES = {
    "elementary_1d",
    "synchronous_2d",
    "agent_2d",
    "sandpile_2d",
    "continuous_2d",
}

VALID_BOUNDARIES = {"wrap", "constant", "reflect", "absorb"}

VALID_INITIAL_MODES = {
    "zeros",
    "single_center",
    "center_seed",
    "random",
    "custom",
}


@dataclass
class GridConfig:
    dimensions: int = 2
    shape: Tuple[int, int] = (100, 100)
    boundary: str = "wrap"
    initial_mode: str = "zeros"
    probabilities: Optional[List[float]] = None
    seed: Optional[int] = None


@dataclass
class NeighborhoodConfig:
    type: str = "moore"
    radius: int = 1


@dataclass
class AutomatonConfig:
    name: str
    engine_type: str
    grid: GridConfig
    states: Dict[str, int] = field(default_factory=lambda: {"DEAD": 0, "ALIVE": 1})
    colors: Dict[int, List[int]] = field(default_factory=dict)
    neighborhood: NeighborhoodConfig = field(default_factory=NeighborhoodConfig)
    rules: Union[List[Dict[str, Any]], Dict[str, Any]] = field(default_factory=list)
    agents: List[Dict[str, Any]] = field(default_factory=list)
    custom_initial_cells: List[List[Union[int, float]]] = field(default_factory=list)
    rule_number: int = 30
    target_pattern: Optional[List[List[int]]] = None
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return self.grid.shape[0]

    @property
    def cols(self) -> int:
        return self.grid.shape[1]

    @property
    def state_id_to_name(self) -> Dict[int, str]:
        return {v: k for k, v in self.states.items()}


class AutomatonConfigParser:
    """
    Loads a .json file or dictionary, validates required fields, converts state keys,
    generates default RGB color palettes, and constructs the initial numpy.ndarray.
    """

    @staticmethod
    def load_file(filepath: str) -> AutomatonConfig:
        """Loads and validates a JSON configuration file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        return AutomatonConfigParser.parse_dict(raw_data)

    @staticmethod
    def load_json_string(json_str: str) -> AutomatonConfig:
        """Parses and validates a JSON string."""
        raw_data = json.loads(json_str)
        return AutomatonConfigParser.parse_dict(raw_data)

    @staticmethod
    def parse_dict(data: Dict[str, Any]) -> AutomatonConfig:
        """Validates raw dictionary and returns a structured AutomatonConfig."""
        if not isinstance(data, dict):
            raise ValueError("Configuration root must be a JSON object.")

        name = data.get("name", "Unnamed Automaton")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Field 'name' must be a non-empty string.")

        engine_type = data.get("engine_type", "").strip().lower()
        if engine_type not in VALID_ENGINE_TYPES:
            raise ValueError(
                f"Invalid 'engine_type': '{engine_type}'. "
                f"Must be one of {sorted(VALID_ENGINE_TYPES)}."
            )

        # Validate grid section
        grid_raw = data.get("grid")
        if not isinstance(grid_raw, dict):
            raise ValueError("Missing or invalid 'grid' object in JSON.")

        dimensions = int(grid_raw.get("dimensions", 2))
        shape_raw = grid_raw.get("shape", [100, 100])
        if (
            not isinstance(shape_raw, (list, tuple))
            or len(shape_raw) not in (1, 2)
            or not all(isinstance(x, int) and x > 0 for x in shape_raw)
        ):
            raise ValueError("'grid.shape' must be a list of 1 or 2 positive integers [H, W].")

        if len(shape_raw) == 1:
            shape = (int(shape_raw[0]), int(shape_raw[0]))
        else:
            shape = (int(shape_raw[0]), int(shape_raw[1]))

        boundary = str(grid_raw.get("boundary", "wrap")).strip().lower()
        if boundary not in VALID_BOUNDARIES:
            raise ValueError(f"Invalid 'grid.boundary': '{boundary}'. Must be one of {sorted(VALID_BOUNDARIES)}.")

        initial_mode = str(grid_raw.get("initial_mode", "zeros")).strip().lower()
        if initial_mode not in VALID_INITIAL_MODES:
            raise ValueError(
                f"Invalid 'grid.initial_mode': '{initial_mode}'. "
                f"Must be one of {sorted(VALID_INITIAL_MODES)}."
            )

        probabilities = grid_raw.get("probabilities", data.get("probabilities"))
        seed = grid_raw.get("seed", data.get("seed"))

        grid_cfg = GridConfig(
            dimensions=dimensions,
            shape=shape,
            boundary=boundary,
            initial_mode=initial_mode,
            probabilities=probabilities,
            seed=seed,
        )

        # Validate states
        states_raw = data.get("states", {"OFF": 0, "ON": 1})
        if not isinstance(states_raw, dict) or len(states_raw) == 0:
            raise ValueError("'states' must be a non-empty dictionary mapping state names to integer IDs.")

        states: Dict[str, int] = {}
        for k, v in states_raw.items():
            states[str(k)] = int(v)

        # Validate or generate RGB colors
        colors_raw = data.get("colors")
        colors: Dict[int, List[int]] = {}
        if isinstance(colors_raw, dict) and len(colors_raw) > 0:
            for k, rgb in colors_raw.items():
                state_id = int(k)
                if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
                    colors[state_id] = [max(0, min(255, int(c))) for c in rgb]
                elif isinstance(rgb, str) and rgb.startswith("#") and len(rgb) == 7:
                    colors[state_id] = [int(rgb[1:3], 16), int(rgb[3:5], 16), int(rgb[5:7], 16)]
                else:
                    raise ValueError(f"Color for state '{k}' must be an [R, G, B] list of 3 integers.")
        else:
            colors = AutomatonConfigParser._generate_default_colors(states, engine_type)

        # Ensure every defined state has a color entry
        default_Generated = AutomatonConfigParser._generate_default_colors(states, engine_type)
        for state_id in states.values():
            if state_id not in colors:
                colors[state_id] = default_Generated.get(state_id, [200, 200, 200])

        # Validate neighborhood
        neigh_raw = data.get("neighborhood", {"type": "moore", "radius": 1})
        if not isinstance(neigh_raw, dict):
            neigh_raw = {"type": "moore", "radius": 1}
        neigh_type = str(neigh_raw.get("type", "moore")).strip().lower()
        neigh_radius = int(neigh_raw.get("radius", 1))
        neighborhood_cfg = NeighborhoodConfig(type=neigh_type, radius=neigh_radius)

        # Rules
        rules_raw = data.get("rules", [])
        rule_number = int(data.get("rule_number", 30))
        if isinstance(rules_raw, dict) and "rule_number" in rules_raw:
            rule_number = int(rules_raw["rule_number"])
        elif isinstance(rules_raw, list):
            for r in rules_raw:
                if isinstance(r, dict) and "rule_number" in r:
                    rule_number = int(r["rule_number"])

        # Agents
        agents_raw = data.get("agents", [])
        if not isinstance(agents_raw, list):
            raise ValueError("'agents' must be a list of agent dictionaries.")

        # Custom initial cells
        custom_cells_raw = data.get("custom_initial_cells", [])
        if not isinstance(custom_cells_raw, list):
            raise ValueError("'custom_initial_cells' must be a list of [row, col, state] triples.")

        target_pattern = data.get("target_pattern")

        return AutomatonConfig(
            name=name,
            engine_type=engine_type,
            grid=grid_cfg,
            states=states,
            colors=colors,
            neighborhood=neighborhood_cfg,
            rules=rules_raw,
            agents=agents_raw,
            custom_initial_cells=custom_cells_raw,
            rule_number=rule_number,
            target_pattern=target_pattern,
            raw_config=data,
        )

    @staticmethod
    def _generate_default_colors(states: Dict[str, int], engine_type: str) -> Dict[int, List[int]]:
        """Generates default RGB palettes if 'colors' is omitted."""
        sorted_ids = sorted(set(states.values()))
        palette: Dict[int, List[int]] = {}
        n = max(1, len(sorted_ids))

        for idx, state_id in enumerate(sorted_ids):
            if state_id == 0:
                palette[state_id] = [20, 20, 28]
            elif n == 2 and state_id == 1:
                palette[state_id] = [0, 230, 180]
            else:
                hue = (idx / n) % 1.0
                palette[state_id] = AutomatonConfigParser._hsv_to_rgb(hue, 0.85, 0.95)
        return palette

    @staticmethod
    def _hsv_to_rgb(h: float, s: float, v: float) -> List[int]:
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        return [int(r * 255), int(g * 255), int(b * 255)]

    @staticmethod
    def generate_initial_grid(config: AutomatonConfig) -> np.ndarray:
        """
        Generates the initial numpy.ndarray based on grid.initial_mode:
        - "zeros": All 0s.
        - "single_center" / "center_seed": All 0s with the center cell (or top-center for 1D) set to 1
          (and applies custom_initial_cells if provided for center_seed).
        - "random": Random choice across defined states using optional "probabilities".
        - "custom": Zero grid populated with [row, col, state] coordinates from "custom_initial_cells".
        """
        rows, cols = config.grid.shape
        dtype = np.float64 if config.engine_type == "continuous_2d" else np.int32
        grid = np.zeros((rows, cols), dtype=dtype)
        mode = config.grid.initial_mode

        if mode == "zeros":
            return grid

        elif mode in ("single_center", "center_seed"):
            if mode == "center_seed" and len(config.custom_initial_cells) > 0:
                for item in config.custom_initial_cells:
                    if len(item) >= 3:
                        r, c, val = int(item[0]), int(item[1]), item[2]
                        if 0 <= r < rows and 0 <= c < cols:
                            grid[r, c] = val
                return grid

            if config.engine_type == "elementary_1d":
                grid[0, cols // 2] = 1
            elif config.engine_type == "sandpile_2d":
                # Check if custom center value is specified in raw_config or custom_initial_cells
                center_val = config.raw_config.get("grid", {}).get("center_value", 256)
                grid[rows // 2, cols // 2] = center_val
            elif config.engine_type == "continuous_2d":
                center_val = float(config.raw_config.get("grid", {}).get("center_value", 100.0))
                grid[rows // 2, cols // 2] = center_val
            else:
                state_ids = sorted(config.states.values())
                active_val = state_ids[1] if len(state_ids) > 1 else 1
                grid[rows // 2, cols // 2] = active_val
            return grid

        elif mode == "random":
            rng = np.random.default_rng(config.grid.seed)
            state_ids = sorted(config.states.values())
            probs = config.grid.probabilities
            if probs is not None:
                if len(probs) != len(state_ids):
                    raise ValueError(
                        f"Length of 'probabilities' ({len(probs)}) must match number of states ({len(state_ids)})."
                    )
                probs_arr = np.array(probs, dtype=np.float64)
                probs_arr = probs_arr / np.sum(probs_arr)
            else:
                probs_arr = None

            if config.engine_type == "elementary_1d":
                grid[0, :] = rng.choice(state_ids, size=cols, p=probs_arr)
            else:
                grid = rng.choice(state_ids, size=(rows, cols), p=probs_arr).astype(dtype)
            return grid

        elif mode == "custom":
            for item in config.custom_initial_cells:
                if len(item) >= 3:
                    r, c, val = int(item[0]), int(item[1]), item[2]
                    if 0 <= r < rows and 0 <= c < cols:
                        grid[r, c] = val
            return grid

        return grid


# Module-level convenience aliases
load_config_from_file = AutomatonConfigParser.load_file
load_config_from_string = AutomatonConfigParser.load_json_string
validate_config = AutomatonConfigParser.parse_dict
generate_initial_grid = AutomatonConfigParser.generate_initial_grid
