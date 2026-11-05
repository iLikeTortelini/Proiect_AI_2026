"""
Module 4: Graphical User Interface (gui.py).
Implements CelLabApp divided into three panels:
- Left Panel: Configuration & Presets, editable JSON editor with "Apply / Validate JSON", N / X / FPS inputs
- Center Panel: High-performance Grid Viewport (PIL.ImageTk NEAREST), Iteration Timeline Slider & Playback Controls
- Right Panel: 5 Questions Analytical Dashboard with Q1 Jump to Step N and Q3 sub-matrix pattern input
"""

from __future__ import annotations

import glob
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple, Union
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
        self.var_iteration = tk.IntVar(value=0)
        self.var_pattern_input = tk.StringVar(value="1,1; 1,1")

        self.drag_draw_val: Optional[Union[int, float]] = None
        self._is_scrubbing_slider: bool = False

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
            width=8,
            bg="#313244",
            fg="#cdd6f4",
            insertbackground="#ffffff",
            font=("Consolas", 10),
            command=self._on_target_n_changed,
        )
        self.spin_n.pack(side=tk.RIGHT)
        self.spin_n.bind("<Return>", lambda e: self._on_target_n_changed())

        # Target Cell State (X)
        row_x = ttk.Frame(params_box, style="Panel.TFrame")
        row_x.pack(fill=tk.X, pady=3)
        ttk.Label(row_x, text="Target Cell State (X):", style="Panel.TLabel").pack(side=tk.LEFT)
        self.spin_x = tk.Spinbox(
            row_x,
            from_=0,
            to=255,
            textvariable=self.var_target_state_x,
            width=8,
            bg="#313244",
            fg="#cdd6f4",
            insertbackground="#ffffff",
            font=("Consolas", 10),
        )
        self.spin_x.pack(side=tk.RIGHT)

        # Speed (FPS) slider
        row_fps = ttk.Frame(params_box, style="Panel.TFrame")
        row_fps.pack(fill=tk.X, pady=(6, 2))
        self.lbl_fps = ttk.Label(row_fps, text=f"Simulation Speed: {self.var_fps.get()} FPS", style="Panel.TLabel")
        self.lbl_fps.pack(anchor=tk.W)

        self.slider_fps = tk.Scale(
            row_fps,
            from_=1,
            to=60,
            orient=tk.HORIZONTAL,
            variable=self.var_fps,
            bg="#1e1e2e",
            fg="#cdd6f4",
            activebackground="#89b4fa",
            troughcolor="#11111b",
            highlightthickness=0,
            command=self._on_fps_changed,
        )
        self.slider_fps.pack(fill=tk.X)

        # =====================================================================
        # CENTER PANEL: Grid Viewport Canvas & Playback Bar
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
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)

        # Playback Control Bar (packed side=tk.BOTTOM first so it's at the very bottom)
        playback_bar = ttk.Frame(center_panel, style="Panel.TFrame", padding="4 6 4 4")
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

        self.btn_set_gen0 = ttk.Button(
            playback_bar, text="📌 Set as Gen 0", command=self.set_current_as_gen0
        )
        self.btn_set_gen0.pack(side=tk.LEFT, padx=4)

        ttk.Label(
            playback_bar,
            text="(Click & drag to draw cells)",
            style="Panel.TLabel",
            font=("Segoe UI", 8, "italic"),
        ).pack(side=tk.RIGHT, padx=4)

        # Iteration Timeline Bar (packed side=tk.BOTTOM second so it sits directly above playback_bar)
        timeline_bar = ttk.Frame(center_panel, style="Panel.TFrame", padding="4 6 4 2")
        timeline_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.lbl_iteration_slider = ttk.Label(
            timeline_bar,
            text="Iteration: 0 / 100",
            style="Panel.TLabel",
            font=("Segoe UI", 9, "bold"),
            width=20,
        )
        self.lbl_iteration_slider.pack(side=tk.LEFT, padx=(4, 6))

        self.slider_iteration = tk.Scale(
            timeline_bar,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            resolution=1,
            bg="#1e1e2e",
            fg="#cdd6f4",
            activebackground="#89b4fa",
            troughcolor="#11111b",
            highlightthickness=0,
            showvalue=False,
            command=self._on_iteration_scrub,
        )
        self.slider_iteration.bind("<Button-1>", lambda e: self.stop_simulation())
        self.slider_iteration.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

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

        # Scrollable container for question result cards
        results_canvas = tk.Canvas(right_panel, bg="#1e1e2e", highlightthickness=0)
        results_scrollbar = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=results_canvas.yview)
        self.results_inner = ttk.Frame(results_canvas, style="Panel.TFrame")

        self.results_inner.bind(
            "<Configure>",
            lambda e: results_canvas.configure(scrollregion=results_canvas.bbox("all")),
        )
        results_canvas.create_window((0, 0), window=self.results_inner, anchor="nw")
        results_canvas.configure(yscrollcommand=results_scrollbar.set)

        results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_analysis_cards()

    def _build_analysis_cards(self) -> None:
        """Constructs individual cards for Questions 1 to 5."""
        # Q1 Card
        card_q1 = self._create_card("Q1: State After N Iterations")
        self.lbl_q1 = ttk.Label(card_q1, text="Click 'Run Full Analysis' to compute state at N.", wraplength=350, style="Panel.TLabel")
        self.lbl_q1.pack(anchor=tk.W, pady=(0, 4))
        self.btn_jump_n = ttk.Button(card_q1, text="➡ Jump Viewport to Step N", command=self.jump_viewport_to_n)
        self.btn_jump_n.pack(fill=tk.X)

        # Q2 Card
        card_q2 = self._create_card("Q2: When Does System Stabilize?")
        self.lbl_q2 = ttk.Label(card_q2, text="Awaiting analysis...", wraplength=350, style="Panel.TLabel")
        self.lbl_q2.pack(anchor=tk.W)

        # Q3 Card
        card_q3 = self._create_card("Q3: Does Target Configuration Occur?")
        row_pat = ttk.Frame(card_q3, style="Panel.TFrame")
        row_pat.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(row_pat, text="Sub-matrix:", style="Panel.TLabel").pack(side=tk.LEFT)
        self.ent_pattern = ttk.Entry(row_pat, textvariable=self.var_pattern_input)
        self.ent_pattern.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        ttk.Label(
            card_q3,
            text="Format e.g. '1,1; 1,1' for a 2x2 block.",
            font=("Segoe UI", 8, "italic"),
            foreground="#6c7086",
            style="Panel.TLabel",
        ).pack(anchor=tk.W, pady=(0, 4))

        self.lbl_q3 = ttk.Label(card_q3, text="Awaiting analysis...", wraplength=350, style="Panel.TLabel")
        self.lbl_q3.pack(anchor=tk.W)

        # Q4 Card
        card_q4 = self._create_card("Q4: Cell Counts of State X After N Steps")
        self.lbl_q4 = ttk.Label(card_q4, text="Awaiting analysis...", wraplength=350, style="Panel.TLabel")
        self.lbl_q4.pack(anchor=tk.W, pady=(0, 4))

        self.txt_q4_breakdown = tk.Text(
            card_q4,
            height=4,
            bg="#11111b",
            fg="#cdd6f4",
            font=("Consolas", 8),
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.txt_q4_breakdown.pack(fill=tk.X)

        # Q5 Card
        card_q5 = self._create_card("Q5: Does Simulation Enter a Cycle?")
        self.lbl_q5 = ttk.Label(card_q5, text="Awaiting analysis...", wraplength=350, style="Panel.TLabel")
        self.lbl_q5.pack(anchor=tk.W)

    def _create_card(self, title: str) -> tk.LabelFrame:
        card = tk.LabelFrame(
            self.results_inner,
            text=f" {title} ",
            bg="#1e1e2e",
            fg="#89b4fa",
            font=("Segoe UI", 9, "bold"),
            padx=8,
            pady=8,
        )
        card.pack(fill=tk.X, expand=True, pady=4, padx=2)
        return card

    # =========================================================================
    # Configuration Loading & Preset Discovery
    # =========================================================================

    def _discover_and_load_presets(self, initial_config_path: Optional[str]) -> None:
        """Finds preset .json files in cellab/presets and initializes dropdown."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        presets_dir = os.path.join(current_dir, "presets")

        files: List[str] = []
        if os.path.exists(presets_dir):
            files = sorted(glob.glob(os.path.join(presets_dir, "*.json")))

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

        if config.target_pattern:
            rows_str = "; ".join(",".join(str(x) for x in r) for r in config.target_pattern)
            self.var_pattern_input.set(rows_str)

        self.update_status_and_viewport(update_slider=True)

    def _build_color_lut(self, config: AutomatonConfig) -> None:
        max_state = max(list(config.colors.keys()) + list(config.states.values()) + [1])
        lut_size = max(2, max_state + 1)
        self.color_lut = np.zeros((lut_size, 3), dtype=np.uint8)
        for state_id, rgb in config.colors.items():
            if 0 <= state_id < lut_size:
                self.color_lut[state_id] = rgb

    # =========================================================================
    # Playback Controls & Timeline Slider
    # =========================================================================

    def play_simulation(self) -> None:
        """Starts non-blocking root.after timer that continuously calls step_forward()."""
        if not self.controller or self.is_playing:
            return

        try:
            target_n = max(1, int(self.var_target_n.get()))
        except Exception:
            target_n = 100

        # If user is already at or past target_n and hits Play, advance target_n by 100
        if self.controller.engine.generation >= target_n:
            target_n = self.controller.engine.generation + 100
            self.var_target_n.set(target_n)

        self.is_playing = True
        self._schedule_next_tick()

    def _schedule_next_tick(self) -> None:
        if not self.is_playing or not self.controller:
            return

        try:
            target_n = max(1, int(self.var_target_n.get()))
        except Exception:
            target_n = 100

        if self.controller.engine.generation >= target_n:
            self.stop_simulation()
            return

        self.controller.step_forward()
        self.update_status_and_viewport(update_slider=True)

        if self.controller.engine.generation >= target_n:
            self.stop_simulation()
            return

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
        self.update_status_and_viewport(update_slider=True)

    def step_back(self) -> None:
        """Pauses playback (if running) and reverts 1 generation."""
        self.stop_simulation()
        if not self.controller:
            return
        self.controller.step_backward()
        self.update_status_and_viewport(update_slider=True)

    def reset_simulation(self) -> None:
        """Stops playback, clears future history, and restores generation 0."""
        self.stop_simulation()
        if not self.controller:
            return
        self.controller.reset()
        self.update_status_and_viewport(update_slider=True)

    def set_current_as_gen0(self) -> None:
        """Sets the current grid state as the new initial state (Gen 0) for future runs."""
        if not self.controller:
            return
        self.stop_simulation()
        self.controller.set_current_as_initial()
        self.update_status_and_viewport(update_slider=True)
        messagebox.showinfo(
            "State Saved as Initial",
            "Current pattern has been set as Generation 0!\nPressing Reset or Play will now run from this state.",
        )

    def _on_fps_changed(self, val: str) -> None:
        fps_int = max(1, int(float(val)))
        self.var_fps.set(fps_int)
        self.lbl_fps.config(text=f"Simulation Speed: {fps_int} FPS")

    def _on_target_n_changed(self) -> None:
        try:
            val = int(self.var_target_n.get())
        except Exception:
            val = 100
        if self.controller:
            self.update_status_and_viewport(update_slider=False)

    def _on_iteration_scrub(self, val_str: str) -> None:
        """Called when user scrubs the generation timeline slider."""
        if self._is_scrubbing_slider or self.is_playing or not self.controller or not self.config:
            return
        try:
            target_gen = int(round(float(val_str)))
        except (ValueError, TypeError):
            return

        if target_gen != self.controller.engine.generation:
            self.controller.seek_generation(target_gen)
            self.update_status_and_viewport(update_slider=False)

    # =========================================================================
    # Viewport Rendering & Interaction
    # =========================================================================

    def update_status_and_viewport(self, update_slider: bool = True) -> None:
        if not self.controller or not self.config:
            return
        eng = self.controller.engine
        active_cells = int(np.count_nonzero(eng.grid > 0))
        total_cells = int(eng.grid.size)

        self.lbl_status_header.config(
            text=f"{self.config.name}  |  Generation: {eng.generation}  |  Active Cells: {active_cells:,} / {total_cells:,}"
        )
        self.btn_back.config(state=tk.NORMAL if eng.generation > 0 else tk.DISABLED)

        # Update Timeline Slider bounds only when changed to avoid interrupting mouse drag
        max_n = max(self.var_target_n.get(), self.controller.max_simulated_gen, eng.generation, 100)
        if getattr(self, "_last_slider_max", None) != max_n:
            self._last_slider_max = max_n
            self.slider_iteration.config(to=max_n)

        if update_slider:
            self._is_scrubbing_slider = True
            try:
                self.slider_iteration.set(eng.generation)
            finally:
                self._is_scrubbing_slider = False

        self.lbl_iteration_slider.config(text=f"Iteration: {eng.generation} / {max_n}")

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
            # Fallback when Pillow is not installed: scale using NumPy repeat so it fills the viewport
            scale_int = max(1, int(min(cw / cols, ch / rows)))
            scaled_matrix = np.repeat(np.repeat(rgb_matrix, scale_int, axis=0), scale_int, axis=1)
            sh, sw, _ = scaled_matrix.shape
            ppm_header = f"P6 {sw} {sh} 255\n".encode("ascii")
            self.photo_image = tk.PhotoImage(data=ppm_header + scaled_matrix.tobytes())
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self.photo_image, anchor=tk.CENTER)

    def _canvas_to_grid_coords(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Maps canvas pixel (x, y) to matrix cell (row, col)."""
        if not self.controller:
            return None
        rows, cols = self.controller.engine.grid.shape
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return None

        scale = min(cw / cols, ch / rows)
        if scale <= 0:
            return None
        gw = cols * scale
        gh = rows * scale
        ox = (cw - gw) / 2.0
        oy = (ch - gh) / 2.0

        c = int((x - ox) / scale)
        r = int((y - oy) / scale)
        if 0 <= r < rows and 0 <= c < cols:
            return r, c
        return None

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Toggles cell state and initiates draw value for dragging."""
        if not self.controller or not self.config:
            return
        coords = self._canvas_to_grid_coords(event.x, event.y)
        if coords is None:
            return
        r, c = coords
        eng = self.controller.engine

        if self.config.engine_type == "sandpile_2d":
            eng.grid[r, c] += 4
            self.drag_draw_val = eng.grid[r, c]
        elif self.config.engine_type == "continuous_2d":
            eng.grid[r, c] = 100.0
            self.drag_draw_val = 100.0
        else:
            state_ids = sorted(self.config.states.values())
            curr = int(eng.grid[r, c])
            next_idx = (state_ids.index(curr) + 1) % len(state_ids) if curr in state_ids else 0
            new_val = state_ids[next_idx]
            eng.grid[r, c] = new_val
            self.drag_draw_val = new_val

        self.controller.on_grid_manually_edited()
        self.update_status_and_viewport(update_slider=True)

    def _on_canvas_drag(self, event: tk.Event) -> None:
        """Allows drawing strokes or erasing cells continuously by dragging mouse."""
        if not self.controller or not self.config or self.drag_draw_val is None:
            return
        coords = self._canvas_to_grid_coords(event.x, event.y)
        if coords is None:
            return
        r, c = coords
        eng = self.controller.engine
        if eng.grid[r, c] != self.drag_draw_val:
            eng.grid[r, c] = self.drag_draw_val
            self.controller.on_grid_manually_edited()
            self.update_status_and_viewport(update_slider=True)

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
        self.update_status_and_viewport(update_slider=True)


# Backwards-compatible alias and launcher
CelLabGUI = CelLabApp


def launch_gui(initial_config: Optional[str] = None) -> None:
    root = tk.Tk()
    app = CelLabApp(root, initial_config_path=initial_config)
    root.mainloop()
