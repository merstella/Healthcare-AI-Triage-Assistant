#!/usr/bin/env python3
"""Train triage model and save artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_ai_agent.inference import save_model_bundle
from healthcare_ai_agent.modeling import train_model
from healthcare_ai_agent.preprocessing import load_and_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train healthcare triage classifier.")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the source triage CSV (Kaggle emergency-service-triage-application data.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Output folder for model and metrics.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Holdout ratio for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for split/model.",
    )
    parser.add_argument(
        "--emergency-recall-target",
        type=float,
        default=0.95,
        help="Minimum emergency recall target used during threshold search.",
    )
    parser.add_argument(
        "--threshold-grid-size",
        type=int,
        default=201,
        help="Number of threshold candidates between 0 and 1.",
    )
    parser.add_argument(
        "--threshold-tuning-cv-folds",
        type=int,
        default=5,
        help="Number of folds for OOF threshold tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_and_preprocess(args.data_path)
    train_output = train_model(
        frame,
        test_size=args.test_size,
        random_state=args.random_state,
        emergency_recall_target=args.emergency_recall_target,
        threshold_grid_size=args.threshold_grid_size,
        threshold_tuning_cv_folds=args.threshold_tuning_cv_folds,
    )

    model_path = output_dir / "triage_model.joblib"
    metrics_path = output_dir / "metrics.json"

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(Path(args.data_path).resolve()),
        "row_count_after_preprocessing": len(frame),
        "random_state": args.random_state,
        "test_size": args.test_size,
        "threshold_tuning_cv_folds": args.threshold_tuning_cv_folds,
        "decision_policy": {
            "type": "emergency_threshold_policy",
            "emergency_threshold": train_output["threshold_tuning"]["best_threshold"],
            "target_emergency_recall": args.emergency_recall_target,
            "constraint_met_on_tuning_split": train_output["threshold_tuning"]["constraint_met"],
        },
    }
    save_model_bundle(train_output["model"], model_path, metadata=metadata)

    serializable_metrics = {
        "x_train_size": train_output["x_train_size"],
        "x_test_size": train_output["x_test_size"],
        "x_tuning_size": train_output["x_tuning_size"],
        "train_metrics": train_output["train_metrics"],
        "holdout_metrics": train_output["holdout_metrics"],
        "threshold_tuning": train_output["threshold_tuning"],
    }
    metrics_path.write_text(json.dumps(serializable_metrics, indent=2), encoding="utf-8")

    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(
        "Holdout summary -> "
        f"accuracy={serializable_metrics['holdout_metrics']['accuracy']:.4f}, "
        f"macro_f1={serializable_metrics['holdout_metrics']['macro_f1']:.4f}, "
        f"emergency_recall={serializable_metrics['holdout_metrics']['emergency_recall']:.4f}"
    )
    print(
        "Threshold tuning -> "
        f"best_threshold={serializable_metrics['threshold_tuning']['best_threshold']:.3f}, "
        f"target_recall={serializable_metrics['threshold_tuning']['target_recall']:.2f}, "
        f"constraint_met={serializable_metrics['threshold_tuning']['constraint_met']}"
    )


if __name__ == "__main__":
    main()
