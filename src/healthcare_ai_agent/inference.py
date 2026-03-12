"""Artifact serialization and inference helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from .constants import MODEL_FEATURE_COLUMNS, TRIAGE_LABELS
from .labeling import decode_index
from .modeling import predict_proba_dict, predict_with_emergency_policy


def save_model_bundle(model: Any, output_path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> None:
    bundle = {
        "model": model,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "labels": TRIAGE_LABELS,
        "metadata": dict(metadata or {}),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(model_path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(Path(model_path))
    required_keys = {"model", "feature_columns", "labels", "metadata"}
    missing = required_keys.difference(bundle.keys())
    if missing:
        raise ValueError(f"Invalid model bundle. Missing keys: {sorted(missing)}")
    return bundle


def get_emergency_threshold_from_bundle(bundle: Mapping[str, Any]) -> float | None:
    value = (
        bundle.get("metadata", {})
        .get("decision_policy", {})
        .get("emergency_threshold")
    )
    if value is None:
        return None
    return float(value)


def predict_from_features(bundle: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, float]:
    row = {column: features[column] for column in MODEL_FEATURE_COLUMNS}
    x_row = pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)
    return predict_proba_dict(bundle["model"], x_row)


def predict_label_from_features(
    bundle: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    emergency_threshold: float | None = None,
) -> tuple[str, float | None]:
    row = {column: features[column] for column in MODEL_FEATURE_COLUMNS}
    x_row = pd.DataFrame([row], columns=MODEL_FEATURE_COLUMNS)
    applied_threshold = (
        emergency_threshold
        if emergency_threshold is not None
        else get_emergency_threshold_from_bundle(bundle)
    )
    prediction = predict_with_emergency_policy(
        bundle["model"],
        x_row,
        emergency_threshold=applied_threshold,
    )[0]
    return decode_index(int(prediction)), applied_threshold
