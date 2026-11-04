"""
CelLab - Universal Cellular Automata Simulator & Analyzer.
Entry point launching the GUI with optional configuration parameters.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cellab.gui import launch_gui


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CelLab - Universal Cellular Automata Simulator & Analyzer"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to an initial JSON cellular automaton configuration file.",
    )
    args = parser.parse_args()

    launch_gui(initial_config=args.config)


if __name__ == "__main__":
    main()
