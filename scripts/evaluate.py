#!/usr/bin/env python3
"""Evaluate saved model on a dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_ai_agent.inference import get_emergency_threshold_from_bundle, load_model_bundle
from healthcare_ai_agent.labeling import encode_labels
from healthcare_ai_agent.modeling import evaluate_model
from healthcare_ai_agent.preprocessing import load_and_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an existing triage model.")
    parser.add_argument("--model-path", required=True, help="Path to model bundle (*.joblib)")
    parser.add_argument("--data-path", required=True, help="Path to evaluation CSV")
    parser.add_argument("--output-path", default="", help="Optional JSON output path for metrics")
    parser.add_argument(
        "--emergency-threshold",
        type=float,
        default=None,
        help="Optional override for emergency threshold. If omitted, use model metadata policy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args.model_path)
    frame = load_and_preprocess(args.data_path)

    model = bundle["model"]
    x = frame[bundle["feature_columns"]]
    y = encode_labels(frame["triage_label"].tolist())
    threshold_from_metadata = get_emergency_threshold_from_bundle(bundle)
    emergency_threshold = (
        args.emergency_threshold
        if args.emergency_threshold is not None
        else threshold_from_metadata
    )

    metrics = evaluate_model(
        model,
        x,
        y,
        emergency_threshold=emergency_threshold,
    )
    metrics["applied_emergency_threshold"] = emergency_threshold

    print(json.dumps(metrics, indent=2))

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved evaluation metrics to: {output_path}")


if __name__ == "__main__":
    main()
