"""
Module 4: Graphical User Interface (gui.py).
Implements CelLabApp divided into three panels:
- Left Panel: Configuration & Presets, editable JSON editor with "Apply / Validate JSON", N / X / FPS inputs
- Center Panel: High-performance Grid Viewport (PIL.ImageTk NEAREST) & Playback Bar (Play, Stop, Next, Back, Reset)
- Right Panel: 5 Questions Analytical Dashboard with Q1 Jump to Step N and Q3 sub-matrix pattern input
"""

from __future__ import annotations

import glob
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional
import numpy as np

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from cellab.config_parser import AutomatonConfig, AutomatonConfigParser
from cellab.engines import Agent2DEngine, BaseEngine, create_engine
from cellab.analyzer import SimulationController, parse_pattern_input


class CelLabApp:
    """
    Main 3-Panel Tkinter GUI Application for CelLab.
    """

    def __init__(self, root: tk.Tk, initial_config_path: Optional[str] = None):
        self.root = root
        self.root.title("CelLab - Universal Cellular Automata Simulator & Analyzer")
        self.root.geometry("1440x860")
        self.root.minsize(1160, 720)

        self.config: Optional[AutomatonConfig] = None
        self.controller: Optional[SimulationController] = None
        self.last_analysis_result: Optional[Dict[str, Any]] = None

        self.is_playing: bool = False
        self.after_id: Optional[str] = None
        self.photo_image: Optional[Any] = None

        # Variables for Simulation Parameters
        self.var_target_n = tk.IntVar(value=100)
        self.var_target_state_x = tk.IntVar(value=1)
        self.var_fps = tk.IntVar(value=20)
        self.var_pattern_input = tk.StringVar(value="1,1; 1,1")

        # Lookup table mapping state ID -> [R, G, B]
        self.color_lut: np.ndarray = np.zeros((256, 3), dtype=np.uint8)

        self._configure_styles()
        self._build_three_panel_ui()
        self._discover_and_load_presets(initial_config_path)

    def _configure_styles(self) -> None:
        self.root.configure(bg="#181825")
        style = ttk.Style(self.root)
        style.theme_use("clam")

        bg_main = "#181825"
        bg_panel = "#1e1e2e"
        fg_text = "#cdd6f4"
        accent = "#89b4fa"

        style.configure(".", background=bg_main, foreground=fg_text, font=("Segoe UI", 10))
        style.configure("TFrame", background=bg_main)
        style.configure("Panel.TFrame", background=bg_panel)
        style.configure("TLabel", background=bg_main, foreground=fg_text)
        style.configure("Panel.TLabel", background=bg_panel, foreground=fg_text)
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"), foreground=accent, background=bg_panel)
        style.configure("Status.TLabel", font=("Consolas", 11, "bold"), foreground="#a6e3a1", background=bg_panel)

        style.configure(
            "TButton",
            font=("Segoe UI", 9, "bold"),
            padding=5,
            background="#313244",
            foreground=fg_text,
        )
        style.map("TButton", background=[("active", "#45475a")])

        style.configure("Play.TButton", background="#a6e3a1", foreground="#11111b")
        style.map("Play.TButton", background=[("active", "#94e2d5")])

        style.configure("Stop.TButton", background="#f38ba8", foreground="#11111b")
        style.map("Stop.TButton", background=[("active", "#eba0ac")])

        style.configure("Accent.TButton", background="#89b4fa", foreground="#11111b")
        style.map("Accent.TButton", background=[("active", "#b4befe")])

        style.configure("TCombobox", fieldbackground="#313244", foreground=fg_text)

    def _build_three_panel_ui(self) -> None:
        """Constructs the Left, Center, and Right panels."""
        main_container = ttk.Frame(self.root, padding=8)
        main_container.pack(fill=tk.BOTH, expand=True)

        # =====================================================================
        # LEFT PANEL: Configuration, Presets, Live JSON Editor, Parameters
        # =====================================================================
        left_panel = ttk.Frame(main_container, style="Panel.TFrame", width=360, padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left_panel.pack_propagate(False)

        ttk.Label(left_panel, text="⚙ Configuration & Presets", style="Title.TLabel").pack(
            anchor=tk.W, pady=(0, 8)
        )

        # Preset selection row
        preset_row = ttk.Frame(left_panel, style="Panel.TFrame")
        preset_row.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(preset_row, text="Preset:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(0, 6))
        self.preset_combo = ttk.Combobox(preset_row, state="readonly", width=18)
        self.preset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        self.preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        btn_load_file = ttk.Button(preset_row, text="Load JSON File...", command=self._on_load_json_file)
        btn_load_file.pack(side=tk.RIGHT)

        # Editable JSON text box
        ttk.Label(left_panel, text="Active JSON Specification (Editable):", style="Panel.TLabel").pack(
            anchor=tk.W, pady=(4, 2)
        )

        editor_frame = ttk.Frame(left_panel, style="Panel.TFrame")
        editor_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.json_text = tk.Text(
            editor_frame,
            bg="#11111b",
            fg="#cdd6f4",
            insertbackground="#ffffff",
            font=("Consolas", 9),
            wrap=tk.NONE,
            undo=True,
            height=18,
        )
        scroll_y = ttk.Scrollbar(editor_frame, orient=tk.VERTICAL, command=self.json_text.yview)
        scroll_x = ttk.Scrollbar(editor_frame, orient=tk.HORIZONTAL, command=self.json_text.xview)
        self.json_text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.json_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        btn_apply_json = ttk.Button(
            left_panel,
            text="✔ Apply / Validate JSON",
            style="Accent.TButton",
            command=self.apply_and_validate_json,
        )
        btn_apply_json.pack(fill=tk.X, pady=(0, 10))

        # Simulation Parameters Section
        params_box = tk.LabelFrame(
            left_panel,
            text=" Simulation & Analysis Parameters ",
            bg="#1e1e2e",
            fg="#89b4fa",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=8,
        )
        params_box.pack(fill=tk.X, side=tk.BOTTOM)

        # Target Iterations (N)
        row_n = ttk.Frame(params_box, style="Panel.TFrame")
        row_n.pack(fill=tk.X, pady=3)
        ttk.Label(row_n, text="Target Iterations (N):", style="Panel.TLabel").pack(side=tk.LEFT)
        self.spin_n = tk.Spinbox(
            row_n,
            from_=1,
            to=100000,
            textvariable=self.var_target_n,
            width=10,
            bg="#313244",
            fg="#cdd6f4",
            insertbackground="#ffffff",
            font=("Consolas", 10),
        )
        self.spin_n.pack(side=tk.RIGHT)

        # Target Cell State (X)
        row_x = ttk.Frame(params_box, style="Panel.TFrame")
        row_x.pack(fill=tk.X, pady=3)
        ttk.Label(row_x, text="Target Cell State (X):", style="Panel.TLabel").pack(side=tk.LEFT)
        self.spin_x = tk.Spinbox(
            row_x,
            from_=0,
            to=255,
            textvariable=self.var_target_state_x,
            width=10,
            bg="#313244",
            fg="#cdd6f4",
            insertbackground="#ffffff",
            font=("Consolas", 10),
        )
        self.spin_x.pack(side=tk.RIGHT)

        # Simulation Speed (FPS slider: 1 to 60)
        row_fps = ttk.Frame(params_box, style="Panel.TFrame")
        row_fps.pack(fill=tk.X, pady=(6, 2))
        self.lbl_fps = ttk.Label(row_fps, text="Simulation Speed: 20 FPS", style="Panel.TLabel")
        self.lbl_fps.pack(anchor=tk.W)

        self.fps_slider = ttk.Scale(
            params_box,
            from_=1,
            to=60,
            orient=tk.HORIZONTAL,
            command=self._on_fps_changed,
        )
        self.fps_slider.set(20)
        self.fps_slider.pack(fill=tk.X, pady=(2, 0))

        # =====================================================================
        # CENTER PANEL: Grid Viewport & Playback Bar
        # =====================================================================
        center_panel = ttk.Frame(main_container, style="Panel.TFrame", padding=10)
        center_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        # Status Header
        header_frame = ttk.Frame(center_panel, style="Panel.TFrame")
        header_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))

        self.lbl_status_header = ttk.Label(
            header_frame,
            text="Automaton: None | Generation: 0 | Active Cells: 0",
            style="Status.TLabel",
        )
        self.lbl_status_header.pack(side=tk.LEFT)

        # Viewport Canvas
        canvas_container = tk.Frame(center_panel, bg="#11111b", bd=2, relief=tk.SUNKEN)
        canvas_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, bg="#11111b", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render_viewport())
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Playback Control Bar
        playback_bar = ttk.Frame(center_panel, style="Panel.TFrame", padding="4 10 4 2")
        playback_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_play = ttk.Button(
            playback_bar, text="▶ Play", style="Play.TButton", command=self.play_simulation
        )
        self.btn_play.pack(side=tk.LEFT, padx=4)

        self.btn_stop = ttk.Button(
            playback_bar, text="⏸ Stop", style="Stop.TButton", command=self.stop_simulation
        )
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        self.btn_next = ttk.Button(
            playback_bar, text="⏭ Next", command=self.step_next
        )
        self.btn_next.pack(side=tk.LEFT, padx=4)

        self.btn_back = ttk.Button(
            playback_bar, text="⏮ Back", command=self.step_back
        )
        self.btn_back.pack(side=tk.LEFT, padx=4)

        self.btn_reset = ttk.Button(
            playback_bar, text="🔄 Reset", command=self.reset_simulation
        )
        self.btn_reset.pack(side=tk.LEFT, padx=4)

        ttk.Label(
            playback_bar,
            text="(Click grid cells to toggle)",
            style="Panel.TLabel",
            font=("Segoe UI", 8, "italic"),
        ).pack(side=tk.RIGHT, padx=4)

        # =====================================================================
        # RIGHT PANEL: 5 Questions Analytical Dashboard
        # =====================================================================
        right_panel = ttk.Frame(main_container, style="Panel.TFrame", width=400, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        ttk.Label(right_panel, text="📊 5-Question Analytical Dashboard", style="Title.TLabel").pack(
            anchor=tk.W, pady=(0, 8)
        )

        self.btn_run_analysis = ttk.Button(
            right_panel,
            text="🚀 Run Full Analysis (Up to N Steps)",
            style="Accent.TButton",
            command=self.execute_5_questions_analysis,
        )
        self.btn_run_analysis.pack(fill=tk.X, pady=(0, 10))

        # Q1 Card: State after N iterations
        q1_box = self._create_question_frame(right_panel, "Q1: State After N Iterations")
        self.lbl_q1 = ttk.Label(
            q1_box, text="Click 'Run Full Analysis' to compute state at N.", style="Panel.TLabel", wraplength=355
        )
        self.lbl_q1.pack(anchor=tk.W, pady=(0, 4))
        self.btn_jump_n = ttk.Button(
            q1_box, text="⏩ Jump Viewport to Step N", command=self.jump_viewport_to_n
        )
        self.btn_jump_n.pack(fill=tk.X)

        # Q2 Card: Stability Detection
        q2_box = self._create_question_frame(right_panel, "Q2: When Does System Stabilize?")
        self.lbl_q2 = ttk.Label(
            q2_box, text="Awaiting analysis...", style="Panel.TLabel", wraplength=355
        )
        self.lbl_q2.pack(anchor=tk.W)

        # Q3 Card: Configuration Match
        q3_box = self._create_question_frame(right_panel, "Q3: Does Target Configuration Occur?")
        pat_row = ttk.Frame(q3_box, style="Panel.TFrame")
        pat_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(pat_row, text="Sub-matrix:", style="Panel.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        self.ent_pattern = tk.Entry(
            pat_row,
            textvariable=self.var_pattern_input,
            bg="#11111b",
            fg="#cdd6f4",
            insertbackground="#ffffff",
            font=("Consolas", 9),
        )
        self.ent_pattern.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lbl_q3 = ttk.Label(
            q3_box,
            text="Format e.g. '1,1; 1,1' for a 2x2 block.",
            style="Panel.TLabel",
            wraplength=355,
        )
        self.lbl_q3.pack(anchor=tk.W)

        # Q4 Card: Cell Counts at N
        q4_box = self._create_question_frame(right_panel, "Q4: Cell Counts of State X After N Steps")
        self.lbl_q4 = ttk.Label(
            q4_box, text="Awaiting analysis...", style="Panel.TLabel", wraplength=355
        )
        self.lbl_q4.pack(anchor=tk.W, pady=(0, 3))
        self.txt_q4_breakdown = tk.Text(
            q4_box,
            height=4,
            bg="#11111b",
            fg="#f9e2af",
            font=("Consolas", 9),
            relief=tk.FLAT,
            state=tk.DISABLED,
        )
        self.txt_q4_breakdown.pack(fill=tk.X)

        # Q5 Card: Cycle Detection
        q5_box = self._create_question_frame(right_panel, "Q5: Does Simulation Enter a Cycle?")
        self.lbl_q5 = ttk.Label(
            q5_box, text="Awaiting analysis...", style="Panel.TLabel", wraplength=355
        )
        self.lbl_q5.pack(anchor=tk.W)

    def _create_question_frame(self, parent: ttk.Widget, title: str) -> tk.LabelFrame:
        box = tk.LabelFrame(
            parent,
            text=f" {title} ",
            bg="#1e1e2e",
            fg="#89b4fa",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=6,
        )
        box.pack(fill=tk.X, pady=4)
        return box

    # =========================================================================
    # Preset & JSON Loading
    # =========================================================================

    def _discover_and_load_presets(self, initial_config_path: Optional[str] = None) -> None:
        preset_dir = os.path.join(os.path.dirname(__file__), "presets")
        files = sorted(glob.glob(os.path.join(preset_dir, "*.json")))

        self.preset_paths: Dict[str, str] = {}
        display_names: List[str] = []

        for fpath in files:
            base = os.path.splitext(os.path.basename(fpath))[0]
            label = base.replace("_", " ").title()
            self.preset_paths[label] = fpath
            display_names.append(label)

        self.preset_combo["values"] = display_names

        if initial_config_path and os.path.exists(initial_config_path):
            self.load_json_file(initial_config_path)
        elif display_names:
            # Prefer Game Of Life as default if present, else first preset
            default_idx = 0
            for i, name in enumerate(display_names):
                if "game of life" in name.lower():
                    default_idx = i
                    break
            self.preset_combo.current(default_idx)
            self.load_json_file(self.preset_paths[display_names[default_idx]])

    def _on_preset_selected(self, event: Any = None) -> None:
        selected = self.preset_combo.get()
        if selected in self.preset_paths:
            self.load_json_file(self.preset_paths[selected])

    def _on_load_json_file(self) -> None:
        filepath = filedialog.askopenfilename(
            title="Select Automaton JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
        )
        if filepath:
            self.load_json_file(filepath)

    def load_json_file(self, filepath: str) -> None:
        """Reads a JSON file, populates the live editor, and initializes the simulation."""
        self.stop_simulation()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
            config = AutomatonConfigParser.load_json_string(raw_text)
        except Exception as exc:
            messagebox.showerror("JSON Load Error", f"Could not load '{filepath}':\n{exc}")
            return

        self.json_text.delete("1.0", tk.END)
        self.json_text.insert("1.0", raw_text.strip())
        self._activate_config(config)

    def apply_and_validate_json(self) -> None:
        """Parses and validates the JSON in the text box and reloads the automaton."""
        self.stop_simulation()
        raw_text = self.json_text.get("1.0", tk.END).strip()
        try:
            config = AutomatonConfigParser.load_json_string(raw_text)
        except Exception as exc:
            messagebox.showerror("Validation Error", f"Invalid JSON configuration:\n\n{exc}")
            return

        self._activate_config(config)
        messagebox.showinfo("Validated", f"Successfully loaded '{config.name}' ({config.engine_type}).")

    def _activate_config(self, config: AutomatonConfig) -> None:
        self.config = config
        engine = create_engine(config)
        if self.controller is None:
            self.controller = SimulationController(engine, max_history=2000)
        else:
            self.controller.set_engine(engine)

        self._build_color_lut(config)

        # If config defines a target_pattern, populate the Q3 input box
        if config.target_pattern:
            rows_str = "; ".join(",".join(str(x) for x in r) for r in config.target_pattern)
            self.var_pattern_input.set(rows_str)

        self.update_status_and_viewport()

    def _build_color_lut(self, config: AutomatonConfig) -> None:
        max_state = max(list(config.colors.keys()) + list(config.states.values()) + [1])
        lut_size = max(2, max_state + 1)
        self.color_lut = np.zeros((lut_size, 3), dtype=np.uint8)
        for state_id, rgb in config.colors.items():
            if 0 <= state_id < lut_size:
                self.color_lut[state_id] = rgb

    # =========================================================================
    # Playback Controls (Play, Stop, Next, Back, Reset)
    # =========================================================================

    def play_simulation(self) -> None:
        """Starts non-blocking root.after timer that continuously calls step_forward()."""
        if not self.controller or self.is_playing:
            return
        self.is_playing = True
        self._schedule_next_tick()

    def _schedule_next_tick(self) -> None:
        if not self.is_playing or not self.controller:
            return
        self.controller.step_forward()
        self.update_status_and_viewport()

        fps = max(1, int(self.var_fps.get()))
        delay_ms = max(10, int(1000 / fps))
        self.after_id = self.root.after(delay_ms, self._schedule_next_tick)

    def stop_simulation(self) -> None:
        """Cancels the root.after timer, pausing evolution immediately."""
        self.is_playing = False
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def step_next(self) -> None:
        """Pauses playback (if running) and advances exactly 1 generation."""
        self.stop_simulation()
        if not self.controller:
            return
        self.controller.step_forward()
        self.update_status_and_viewport()

    def step_back(self) -> None:
        """Pauses playback (if running) and reverts 1 generation."""
        self.stop_simulation()
        if not self.controller:
            return
        self.controller.step_backward()
        self.update_status_and_viewport()

    def reset_simulation(self) -> None:
        """Stops playback, clears history, and restores generation 0."""
        self.stop_simulation()
        if not self.controller:
            return
        self.controller.reset()
        self.update_status_and_viewport()

    def _on_fps_changed(self, val: str) -> None:
        fps_int = max(1, int(float(val)))
        self.var_fps.set(fps_int)
        self.lbl_fps.config(text=f"Simulation Speed: {fps_int} FPS")

    # =========================================================================
    # Viewport Rendering & Interaction
    # =========================================================================

    def update_status_and_viewport(self) -> None:
        if not self.controller or not self.config:
            return
        eng = self.controller.engine
        active_cells = int(np.count_nonzero(eng.grid > 0))
        total_cells = int(eng.grid.size)

        self.lbl_status_header.config(
            text=f"{self.config.name}  |  Generation: {eng.generation}  |  Active Cells: {active_cells:,} / {total_cells:,}"
        )
        self.btn_back.config(state=tk.NORMAL if len(self.controller.history) > 0 else tk.DISABLED)
        self.render_viewport()

    def render_viewport(self) -> None:
        """Renders current np.ndarray mapped through the RGB color palette onto tk.Canvas."""
        if not self.controller or not self.config:
            return

        eng = self.controller.engine
        grid = eng.get_display_grid()
        rows, cols = grid.shape

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        if eng.config.engine_type == "continuous_2d":
            # Map continuous values [0, max] to thermal color gradient
            max_val = float(np.max(grid)) if np.max(grid) > 0 else 1.0
            norm = np.clip(grid / max_val, 0.0, 1.0)
            r_chan = np.clip(norm * 2.0, 0.0, 1.0) * 255
            g_chan = np.clip((norm - 0.3) * 1.5, 0.0, 1.0) * 255
            b_chan = np.clip(0.15 + (1.0 - norm) * 0.5, 0.0, 1.0) * (norm > 0.001) * 200
            rgb_matrix = np.stack([r_chan, g_chan, b_chan], axis=-1).astype(np.uint8)
        else:
            idx_grid = np.clip(grid.astype(np.int32), 0, len(self.color_lut) - 1)
            rgb_matrix = self.color_lut[idx_grid].copy()

            # Highlight agents (e.g., Langton's Ant) in bright red/orange
            if isinstance(eng, Agent2DEngine):
                for ag in eng.agents:
                    ar, ac = ag["row"], ag["col"]
                    if 0 <= ar < rows and 0 <= ac < cols:
                        rgb_matrix[ar, ac] = [255, 60, 60]

        if PIL_AVAILABLE:
            scale = min(cw / cols, ch / rows)
            target_w = max(1, int(cols * scale))
            target_h = max(1, int(rows * scale))
            pil_img = Image.fromarray(rgb_matrix, mode="RGB").resize(
                (target_w, target_h), resample=Image.Resampling.NEAREST
            )
            self.photo_image = ImageTk.PhotoImage(pil_img)
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self.photo_image, anchor=tk.CENTER)
        else:
            ppm_header = f"P6 {cols} {rows} 255\n".encode("ascii")
            self.photo_image = tk.PhotoImage(data=ppm_header + rgb_matrix.tobytes())
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self.photo_image, anchor=tk.CENTER)

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Toggles or increments cell state when user clicks on the grid viewport."""
        if not self.controller or not self.config:
            return
        eng = self.controller.engine
        rows, cols = eng.grid.shape
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        scale = min(cw / cols, ch / rows)
        gw = cols * scale
        gh = rows * scale
        ox = (cw - gw) / 2.0
        oy = (ch - gh) / 2.0

        c = int((event.x - ox) / scale)
        r = int((event.y - oy) / scale)

        if 0 <= r < rows and 0 <= c < cols:
            if self.config.engine_type == "sandpile_2d":
                eng.grid[r, c] += 4
            elif self.config.engine_type == "continuous_2d":
                eng.grid[r, c] = 100.0
            else:
                state_ids = sorted(self.config.states.values())
                curr = int(eng.grid[r, c])
                next_idx = (state_ids.index(curr) + 1) % len(state_ids) if curr in state_ids else 0
                eng.grid[r, c] = state_ids[next_idx]
            self.update_status_and_viewport()

    # =========================================================================
    # Right Panel: 5-Question Analytical Execution
    # =========================================================================

    def execute_5_questions_analysis(self) -> None:
        """Runs the 5-Question analyzer up to N steps on a cloned engine and displays results."""
        if not self.controller:
            return

        try:
            max_steps = max(1, int(self.var_target_n.get()))
        except Exception:
            max_steps = 100
            self.var_target_n.set(100)

        try:
            target_x = int(self.var_target_state_x.get())
        except Exception:
            target_x = 1
            self.var_target_state_x.set(1)

        pattern_str = self.var_pattern_input.get().strip()
        try:
            pattern_arr = parse_pattern_input(pattern_str) if pattern_str else None
        except Exception as exc:
            messagebox.showerror(
                "Invalid Pattern Format",
                f"Could not parse Q3 sub-matrix pattern '{pattern_str}'.\n"
                f"Use format like '1,1; 1,1' or JSON '[[1,1],[1,1]]'.\n\nDetails: {exc}",
            )
            return

        result = self.controller.run_analysis(
            max_steps=max_steps,
            target_state_id=target_x,
            target_pattern=pattern_arr,
        )
        self.last_analysis_result = result

        # Populate Q1..Q5 UI labels
        self.lbl_q1.config(text=result["q1_state_at_n"]["summary"])
        self.lbl_q2.config(text=result["q2_stability"]["summary"])
        self.lbl_q3.config(text=result["q3_pattern_match"]["summary"])
        self.lbl_q4.config(text=result["q4_cell_counts"]["summary"])

        # Populate Q4 State breakdown text
        breakdown = result["q4_cell_counts"]["breakdown"]
        lines = []
        for s_name, info in breakdown.items():
            lines.append(
                f"{s_name:<10} (ID {info['state_id']}): {info['count']:>6,} cells ({info['percentage']:>5.1f}%)"
            )
        self.txt_q4_breakdown.config(state=tk.NORMAL)
        self.txt_q4_breakdown.delete("1.0", tk.END)
        self.txt_q4_breakdown.insert("1.0", "\n".join(lines))
        self.txt_q4_breakdown.config(state=tk.DISABLED)

        self.lbl_q5.config(text=result["q5_cycle"]["summary"])

    def jump_viewport_to_n(self) -> None:
        """Jumps the active simulation viewport to step N."""
        if not self.controller:
            return
        self.stop_simulation()
        try:
            target_n = max(0, int(self.var_target_n.get()))
        except Exception:
            target_n = 100
        self.controller.jump_to_generation(target_n)
        self.update_status_and_viewport()


# Backwards-compatible alias and launcher
CelLabGUI = CelLabApp


def launch_gui(initial_config: Optional[str] = None) -> None:
    root = tk.Tk()
    app = CelLabApp(root, initial_config_path=initial_config)
    root.mainloop()
