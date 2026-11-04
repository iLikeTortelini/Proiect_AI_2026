# CelLab JSON Configuration Specification

This document details the complete JSON schema used by **CelLab**, describing every top-level key, nested object, permissible values, data types, and engine behaviors.

---

## 1. Top-Level Schema Overview

A CelLab configuration is a standard `.json` file containing the following root properties:

```json
{
  "name": "Brian's Brain",
  "engine_type": "synchronous_2d",
  "grid": {
    "dimensions": 2,
    "shape": [100, 100],
    "boundary": "wrap",
    "initial_mode": "center_seed"
  },
  "states": {
    "OFF": 0,
    "FIRING": 1,
    "DYING": 2
  },
  "colors": {
    "0": [20, 20, 20],
    "1": [0, 200, 255],
    "2": [0, 60, 130]
  },
  "neighborhood": {
    "type": "moore",
    "radius": 1
  },
  "rules": [
    {"from": 1, "to": 2, "condition": {"type": "always"}},
    {"from": 2, "to": 0, "condition": {"type": "always"}},
    {"from": 0, "to": 1, "condition": {"type": "neighbor_count", "state": 1, "op": "==", "value": 2}}
  ],
  "agents": [],
  "custom_initial_cells": [
    [49, 50, 1], [50, 49, 1], [50, 51, 1], [51, 50, 2]
  ],
  "target_pattern": [[1, 1], [1, 1]]
}
```

---

## 2. Field-by-Field Reference

### `name` (string, required)
- **Description**: Human-readable display title for the cellular automaton.
- **Example**: `"Conway's Game of Life"`, `"Wolfram Rule 30"`.

---

### `engine_type` (string, required)
Determines which simulation engine processes the automaton:

| Value | Engine | Description |
|---|---|---|
| `"elementary_1d"` | `Elementary1DEngine` | 1D elementary cellular automata (Wolfram rules 0–255). Renders as a 2D space-time diagram advancing downward row-by-row. |
| `"synchronous_2d"` | `Synchronous2DEngine` | Declarative 2D synchronous CA with customizable neighborhoods, neighbor counts, cyclic transitions, voter models, and probabilistic rules. |
| `"agent_2d"` | `Agent2DEngine` | Agent-based 2D automata (e.g., Langton's Ant, Turmites) tracking moving, turning entities that modify underlying cell states. |
| `"sandpile_2d"` | `Sandpile2DEngine` | Abelian Sandpile Model (Bak-Tang-Wiesenfeld) with simultaneous toppling when cell grains $\ge 4$. |
| `"continuous_2d"` | `Continuous2DEngine` | Continuous floating-point grid running 5-point heat diffusion $\text{grid}_{t+1} = 0.2 \times (\text{grid}_t + \text{north} + \text{south} + \text{east} + \text{west})$. |

---

### `grid` (object, required)
Defines grid geometry, boundary conditions, and initialization behavior.

```json
"grid": {
  "dimensions": 2,
  "shape": [100, 100],
  "boundary": "wrap",
  "initial_mode": "random",
  "probabilities": [0.8, 0.2],
  "seed": 42
}
```

#### Properties:
- **`dimensions`** (`integer`, default `2`):
  - `1`: 1-dimensional array (used by `elementary_1d`).
  - `2`: 2-dimensional grid.
- **`shape`** (`[integer, integer]`, required):
  - `[rows, cols]`: Dimensions of the grid (e.g. `[100, 100]`).
- **`boundary`** (`string`, default `"wrap"`):
  - `"wrap"`: Periodic toroidal boundary. Cells wrap around edges (top connects to bottom, left connects to right).
  - `"constant"` / `"absorb"`: Zero-padded boundary. Cells beyond edges are treated as state 0 (grains/heat fall off edges).
  - `"reflect"`: Symmetric reflection. Edge values are mirrored.
- **`initial_mode`** (`string`, default `"zeros"`):
  - `"zeros"`: Grid starts completely filled with `0`.
  - `"single_center"`: Grid starts as all `0`, with the center cell (or top-center for 1D) set to `1`.
  - `"center_seed"`: Center cell is set to `1`, and any cells listed in `custom_initial_cells` are placed on the grid.
  - `"random"`: Cells are randomly chosen across all states defined in `states`.
  - `"custom"`: All cells start at `0`, and only coordinates declared in `custom_initial_cells` are activated.
- **`probabilities`** (`[float, ...]`, optional):
  - Used when `initial_mode` is `"random"`. An array of probabilities summing to `1.0` whose length matches the number of defined states.
  - *Example*: `[0.75, 0.25]` means 75% chance for state 0 and 25% chance for state 1.
- **`seed`** (`integer`, optional):
  - Fixed pseudorandom number generator seed for reproducible random grids.
- **`center_value`** (`number`, optional):
  - Center cell value for sandpile drops (e.g. `20000`) or heat diffusion sources (e.g. `100.0`).

---

### `states` (object, required)
Maps semantic state names to integer state IDs:

```json
"states": {
  "DEAD": 0,
  "ALIVE": 1
}
```
- Keys must be strings (`"OFF"`, `"FIRING"`, `"TREE"`, `"BURNING"`).
- Values must be non-negative integers (`0`, `1`, `2`, ...). State `0` is typically the background/inactive state.

---

### `colors` (object, optional)
Maps state IDs (as string or int keys) to RGB triples or hex strings.

```json
"colors": {
  "0": [20, 20, 28],
  "1": [0, 230, 180],
  "2": [0, 60, 130]
}
```
- Values can be:
  - `[R, G, B]`: 3 integers between `0` and `255`.
  - `"#RRGGBB"`: Standard hex color string (e.g. `"#00ffcc"`).
- *Default*: If omitted, CelLab automatically generates an HSV color palette.

---

### `neighborhood` (object, optional)
Defines how neighbors are counted for `synchronous_2d`.

```json
"neighborhood": {
  "type": "moore",
  "radius": 1
}
```
- **`type`** (`string`, default `"moore"`):
  - `"moore"`: 8 surrounding neighbors (includes diagonals: N, NE, E, SE, S, SW, W, NW).
  - `"von_neumann"` / `"vn"`: 4 orthogonal neighbors (N, S, E, W).
- **`radius`** (`integer`, default `1`):
  - Distance of neighbors to inspect ($1 = 3\times3$ kernel, $2 = 5\times5$ kernel).

---

### `rules` (array of objects, engine-specific)

#### For `synchronous_2d`:
A list of declarative state transition rules evaluated simultaneously on a double-buffered grid:

```json
{
  "from": 0,
  "to": 1,
  "condition": {
    "type": "neighbor_count",
    "state": 1,
    "op": "==",
    "value": 3
  }
}
```

##### Rule Fields:
- **`from`** (`integer`, `[integer, ...]`, or `"*"`):
  - The cell's current state. If `"*"` or omitted, applies to all states.
- **`to`** (`integer`):
  - The state the cell transitions into if the condition is satisfied.
- **`condition`** (`object`):
  - **`type: "always"`**:
    - Unconditional state change.
    - *Example*: `{"from": 1, "to": 2, "condition": {"type": "always"}}`
  - **`type: "neighbor_count"`**:
    - Evaluates the count of neighbors matching `state`.
    - `state` (`integer`): Which neighbor state to count.
    - `op` (`string`): Comparison operator:
      - `"=="`: Exactly equal to `value`.
      - `"!="`: Not equal to `value`.
      - `">="`: Greater than or equal to `value`.
      - `"<="`: Less than or equal to `value`.
      - `">"`: Strictly greater than `value`.
      - `"<"`: Strictly less than `value`.
      - `"in"`: Contained within a list of values (e.g. `"value": [2, 3]`).
    - `value` (`integer` or `[integer, ...]`): Comparison target.
  - **`type: "cyclic"`**:
    - For Cyclic Automata. Cell advances from $s \to (s + 1) \pmod K$ if at least $T$ neighbors have state $(s + 1) \pmod K$.
    - `num_states` (`integer`): Total number of cycle states ($K$).
    - `threshold` (`integer`): Minimum required neighbors ($T$).
  - **`type: "majority"`**:
    - For Voter / Ising models. Cell adopts the most frequent neighbor state.
  - **`type: "probabilistic"`**:
    - Random stochastic transition with probability `prob` (between `0.0` and `1.0`).
    - Can combine with neighbor conditions (e.g. Forest Fire tree ignition).

---

#### For `elementary_1d`:
- Specified either at the root level via `"rule_number": 30` or inside `rules`:
```json
"rules": [
  {"rule_number": 30}
]
```
- Value is an integer between `0` and `255` (Wolfram rule code).

---

#### For `agent_2d` (Langton's Ant):
- Defines turning and cell state modifications:
```json
"rules": [
  {"from": 0, "to": 1, "turn": "R"},
  {"from": 1, "to": 0, "turn": "L"}
]
```
- **`turn`** (`string`):
  - `"R"`: Turn 90° clockwise / right (+1).
  - `"L"`: Turn 90° counter-clockwise / left (-1).
  - `"U"`: Turn 180° / U-turn (+2).
  - `"N"`: Continue straight without turning (0).

---

### `agents` (array of objects, for `agent_2d`)
Defines the initial state of agents placed on the grid:

```json
"agents": [
  {
    "row": 50,
    "col": 50,
    "dir": 0
  }
]
```
- **`row`** (`integer`): Initial row index.
- **`col`** (`integer`): Initial column index.
- **`dir`** (`integer`): Heading direction:
  - `0`: North / Up ($[-1, 0]$)
  - `1`: East / Right ($[0, +1]$)
  - `2`: South / Down ($[+1, 0]$)
  - `3`: West / Left ($[0, -1]$)

---

### `custom_initial_cells` (array of triples, optional)
Explicitly places cells on the grid during initialization:

```json
"custom_initial_cells": [
  [49, 50, 1],
  [50, 49, 1],
  [50, 51, 1],
  [51, 50, 2]
]
```
- Each item is `[row, col, state_value]`.
- Used when `initial_mode` is `"custom"` or `"center_seed"`.

---

### `target_pattern` (2D array of integers, optional)
Pre-populates the sub-matrix pattern for **Q3 (Configuration Match)** in the analytical solver:

```json
"target_pattern": [
  [1, 1],
  [1, 1]
]
```
- Matches any subgrid of identical dimensions anywhere on the board.

---

## 3. Complete Preset Examples

### Example A: Conway's Game of Life
```json
{
  "name": "Conway's Game of Life",
  "engine_type": "synchronous_2d",
  "grid": {
    "dimensions": 2,
    "shape": [100, 100],
    "boundary": "wrap",
    "initial_mode": "random",
    "probabilities": [0.78, 0.22],
    "seed": 42
  },
  "states": {
    "DEAD": 0,
    "ALIVE": 1
  },
  "colors": {
    "0": [17, 17, 27],
    "1": [0, 255, 204]
  },
  "neighborhood": {
    "type": "moore",
    "radius": 1
  },
  "rules": [
    {"from": 1, "to": 0, "condition": {"type": "neighbor_count", "state": 1, "op": "<", "value": 2}},
    {"from": 1, "to": 0, "condition": {"type": "neighbor_count", "state": 1, "op": ">", "value": 3}},
    {"from": 0, "to": 1, "condition": {"type": "neighbor_count", "state": 1, "op": "==", "value": 3}}
  ],
  "agents": [],
  "custom_initial_cells": []
}
```

### Example B: Langton's Ant
```json
{
  "name": "Langton's Ant",
  "engine_type": "agent_2d",
  "grid": {
    "dimensions": 2,
    "shape": [100, 100],
    "boundary": "wrap",
    "initial_mode": "zeros"
  },
  "states": {
    "WHITE": 0,
    "BLACK": 1
  },
  "colors": {
    "0": [24, 24, 37],
    "1": [205, 214, 244]
  },
  "neighborhood": {
    "type": "von_neumann",
    "radius": 1
  },
  "rules": [
    {"from": 0, "to": 1, "turn": "R"},
    {"from": 1, "to": 0, "turn": "L"}
  ],
  "agents": [
    {"row": 50, "col": 50, "dir": 0}
  ],
  "custom_initial_cells": []
}
```

### Example C: Bak-Tang-Wiesenfeld Sandpile
```json
{
  "name": "Abelian Sandpile (BTW)",
  "engine_type": "sandpile_2d",
  "grid": {
    "dimensions": 2,
    "shape": [101, 101],
    "boundary": "wrap",
    "initial_mode": "custom"
  },
  "states": {
    "EMPTY": 0,
    "ONE": 1,
    "TWO": 2,
    "THREE": 3,
    "UNSTABLE": 4
  },
  "colors": {
    "0": [15, 20, 30],
    "1": [6, 182, 212],
    "2": [59, 130, 246],
    "3": [245, 158, 11],
    "4": [239, 68, 68]
  },
  "neighborhood": {
    "type": "von_neumann",
    "radius": 1
  },
  "rules": [],
  "agents": [],
  "custom_initial_cells": [
    [50, 50, 128]
  ]
}
```
