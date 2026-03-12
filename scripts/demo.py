#!/usr/bin/env python3
"""Run interactive triage demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_ai_agent.cli import run_interactive_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run healthcare triage CLI demo.")
    parser.add_argument(
        "--model-path",
        default="artifacts/triage_model.joblib",
        help="Path to trained model bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_interactive_demo(args.model_path)


if __name__ == "__main__":
    main()

