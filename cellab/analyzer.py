"""
Module 3: History Manager & 5-Question Analyzer (analyzer.py).
Implements SimulationController wrapping an active BaseEngine instance,
managing the 2000-step undo history deque, and answering the 5 analytical questions.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from cellab.engines import BaseEngine, Elementary1DEngine


def parse_pattern_input(pattern_input: Union[str, List[Any], np.ndarray, None]) -> Optional[np.ndarray]:
    """
    Parses a 2D pattern from a string (e.g. '1,1; 1,1' or JSON '[[1,1],[1,1]]'),
    a list of lists, or an existing numpy.ndarray.
    Returns a 2D numpy.ndarray or None if empty.
    """
    if pattern_input is None:
        return None
    if isinstance(pattern_input, np.ndarray):
        if pattern_input.size == 0:
            return None
        return pattern_input.reshape(1, -1) if pattern_input.ndim == 1 else pattern_input

    if isinstance(pattern_input, list):
        if len(pattern_input) == 0:
            return None
        arr = np.array(pattern_input)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    if isinstance(pattern_input, str):
        text = pattern_input.strip()
        if not text:
            return None
        # Try JSON array first
        if text.startswith("["):
            data = json.loads(text)
            arr = np.array(data)
            return arr.reshape(1, -1) if arr.ndim == 1 else arr

        # Parse semicolon/newline separated rows: "1,1; 1,1" or "1 1\n1 1"
        rows_raw = [r.strip() for r in text.replace("\n", ";").split(";") if r.strip()]
        if not rows_raw:
            return None
        matrix = []
        for r_str in rows_raw:
            tokens = [t.strip() for t in r_str.replace(",", " ").split() if t.strip()]
            matrix.append([int(float(t)) for t in tokens])
        arr = np.array(matrix, dtype=np.int32)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr

    return None


def match_subgrid_pattern(grid: np.ndarray, pattern: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Uses np.lib.stride_tricks.sliding_window_view to check if `pattern`
    appears anywhere inside `grid`.
    Returns (row, col) of the first match, or None if not found.
    """
    if pattern is None or pattern.size == 0:
        return None
    if pattern.ndim == 1:
        pattern = pattern.reshape(1, -1)

    ph, pw = pattern.shape
    gh, gw = grid.shape
    if ph > gh or pw > gw:
        return None

    windows = sliding_window_view(grid, (ph, pw))
    if grid.dtype in (np.float32, np.float64):
        matches = np.all(np.isclose(windows, pattern, atol=1e-4), axis=(-2, -1))
    else:
        matches = np.all(windows == pattern, axis=(-2, -1))

    if np.any(matches):
        coords = np.argwhere(matches)[0]
        return int(coords[0]), int(coords[1])
    return None


class SimulationController:
    """
    Wraps an active BaseEngine instance, provides Playback & History stack
    (step_forward, step_backward, reset), and executes the 5-Question Analyzer.
    """

    def __init__(self, engine: BaseEngine, max_history: int = 2000):
        self.engine: BaseEngine = engine
        self.history: Deque[Tuple[int, np.ndarray, Any]] = deque(maxlen=max_history)

    def set_engine(self, engine: BaseEngine) -> None:
        """Replaces the active engine and clears history."""
        self.engine = engine
        self.history.clear()

    def step_forward(self) -> np.ndarray:
        """Pushes current state to self.history, calls engine.step(), and returns new state."""
        self.history.append(
            (
                self.engine.generation,
                self.engine.grid.copy(),
                self.engine.get_agent_state_copy(),
            )
        )
        return self.engine.step()

    def step_backward(self) -> Optional[np.ndarray]:
        """Pops the last state from self.history and restores it into engine (if history is non-empty)."""
        if not self.history:
            return None
        gen_num, grid_snap, agent_snap = self.history.pop()
        self.engine.set_state(grid_snap, gen_num, agent_snap)
        return self.engine.grid.copy()

    def reset(self) -> np.ndarray:
        """Clears self.history and calls engine.reset()."""
        self.history.clear()
        return self.engine.reset()

    def jump_to_generation(self, target_gen: int) -> np.ndarray:
        """
        Advances or resets-and-advances the active engine to `target_gen`,
        recording steps in history so Back continues to work seamlessly.
        """
        target_gen = max(0, int(target_gen))
        if self.engine.generation > target_gen:
            self.reset()
        while self.engine.generation < target_gen:
            self.step_forward()
        return self.engine.grid.copy()

    def run_analysis(
        self,
        max_steps: int,
        target_state_id: int = 1,
        target_pattern: Optional[Union[np.ndarray, str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the 5-Question analytical evaluation on a cloned engine instance
        from generation 0 up to `max_steps` (N) without disrupting the GUI view.

        Answers:
        - Q1: What is the state after N iterations?
        - Q2: When does the system reach a stable state?
        - Q3: Does a particular configuration ever occur?
        - Q4: How many cells of type X exist after N iterations?
        - Q5: Does the simulation enter a cycle?
        """
        max_steps = max(0, int(max_steps))
        target_state_id = int(target_state_id)

        pattern_arr = parse_pattern_input(target_pattern)
        if pattern_arr is None and self.engine.config.target_pattern is not None:
            pattern_arr = parse_pattern_input(self.engine.config.target_pattern)

        cloned: BaseEngine = self.engine.clone()
        cloned.reset()

        # Tracking structures for Q2, Q3, Q5
        def get_analysis_hash(eng: BaseEngine) -> bytes:
            if isinstance(eng, Elementary1DEngine):
                return eng.active_row.tobytes()
            return eng.get_state_hash()

        initial_hash = get_analysis_hash(cloned)
        prev_hash = initial_hash
        seen_hashes: Dict[bytes, int] = {initial_hash: 0}

        stable_generation: Optional[int] = None
        cycle_start_gen: Optional[int] = None
        cycle_period: Optional[int] = None

        pattern_found_gen: Optional[int] = None
        pattern_found_coords: Optional[Tuple[int, int]] = None

        # Check generation 0 for Q3 target pattern
        if pattern_arr is not None:
            match_0 = match_subgrid_pattern(cloned.grid, pattern_arr)
            if match_0 is not None:
                pattern_found_gen = 0
                pattern_found_coords = match_0

        # Simulate up to N steps
        for step_idx in range(1, max_steps + 1):
            cloned.step()
            curr_hash = get_analysis_hash(cloned)

            # Q2: Check consecutive stability
            if stable_generation is None and curr_hash == prev_hash:
                stable_generation = step_idx - 1

            # Q5: Check cycle detection
            if curr_hash in seen_hashes:
                start_gen = seen_hashes[curr_hash]
                period = step_idx - start_gen
                if period == 1:
                    if stable_generation is None:
                        stable_generation = start_gen
                elif period > 1 and cycle_period is None:
                    cycle_start_gen = start_gen
                    cycle_period = period
            else:
                seen_hashes[curr_hash] = step_idx

            prev_hash = curr_hash

            # Q3: Check target pattern match if not yet found
            if pattern_arr is not None and pattern_found_gen is None:
                match_pos = match_subgrid_pattern(cloned.grid, pattern_arr)
                if match_pos is not None:
                    pattern_found_gen = step_idx
                    pattern_found_coords = match_pos

        # Final grid at step N (Q1)
        final_grid = cloned.grid.copy()
        total_cells = int(final_grid.size)

        # Q4: Count cells of target_state_id X and breakdown for all defined states
        target_state_count = int(np.count_nonzero(final_grid == target_state_id))
        state_breakdown: Dict[str, Dict[str, Union[int, float]]] = {}

        for state_name, state_val in self.engine.config.states.items():
            cnt = int(np.count_nonzero(final_grid == state_val))
            pct = round((cnt / max(1, total_cells)) * 100.0, 2)
            state_breakdown[state_name] = {
                "state_id": state_val,
                "count": cnt,
                "percentage": pct,
            }

        # Format human-readable messages for Q1..Q5
        q1_summary = (
            f"Computed state at generation N = {max_steps} "
            f"(Shape: {final_grid.shape[0]}x{final_grid.shape[1]})."
        )

        if stable_generation is not None:
            q2_summary = f"Stabilizes at generation {stable_generation} (no further changes)."
        else:
            q2_summary = f"Did not stabilize within {max_steps} steps"

        if pattern_arr is None:
            q3_summary = "No target pattern specified."
        elif pattern_found_gen is not None:
            q3_summary = (
                f"Pattern ({pattern_arr.shape[0]}x{pattern_arr.shape[1]}) first occurs at "
                f"generation {pattern_found_gen} at (row={pattern_found_coords[0]}, col={pattern_found_coords[1]})."
            )
        else:
            q3_summary = (
                f"Pattern ({pattern_arr.shape[0]}x{pattern_arr.shape[1]}) "
                f"did not occur within {max_steps} steps."
            )

        state_label = self.engine.config.state_id_to_name.get(target_state_id, f"State {target_state_id}")
        target_pct = round((target_state_count / max(1, total_cells)) * 100.0, 2)
        q4_summary = (
            f"State {target_state_id} ({state_label}): {target_state_count:,} cells ({target_pct}%) "
            f"after {max_steps} iterations."
        )

        if cycle_period is not None and cycle_start_gen is not None:
            q5_summary = f"Enters a cycle at generation {cycle_start_gen} with period {cycle_period}"
        elif stable_generation is not None:
            q5_summary = (
                f"Reaches a stable fixed point (period 1) at generation {stable_generation}."
            )
        else:
            q5_summary = f"No cycle detected within {max_steps} steps."

        return {
            "max_steps": max_steps,
            "target_state_id": target_state_id,
            "q1_state_at_n": {
                "generation": max_steps,
                "grid": final_grid,
                "summary": q1_summary,
            },
            "q2_stability": {
                "stabilized": stable_generation is not None,
                "generation": stable_generation,
                "summary": q2_summary,
            },
            "q3_pattern_match": {
                "pattern_specified": pattern_arr is not None,
                "occurred": pattern_found_gen is not None,
                "generation": pattern_found_gen,
                "coordinates": pattern_found_coords,
                "summary": q3_summary,
            },
            "q4_cell_counts": {
                "target_state_id": target_state_id,
                "target_state_name": state_label,
                "target_count": target_state_count,
                "target_percentage": target_pct,
                "total_cells": total_cells,
                "breakdown": state_breakdown,
                "summary": q4_summary,
            },
            "q5_cycle": {
                "entered_cycle": cycle_period is not None,
                "start_generation": cycle_start_gen,
                "period": cycle_period,
                "summary": q5_summary,
            },
        }
