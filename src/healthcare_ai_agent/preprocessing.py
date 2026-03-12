"""Data loading and preprocessing for triage modeling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .constants import (
    ARRIVAL_MODE_MAP,
    INJURY_MAP,
    KTAS_TO_TRIAGE_MAP,
    MENTAL_MAP,
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
    PAIN_MAP,
    RAW_FEATURE_COLUMNS,
    RAW_TARGET_COLUMN,
    SEX_MAP,
)


def bucket_age(age: float) -> str:
    if age < 25:
        return "Young"
    if age < 45:
        return "Adult"
    if age < 60:
        return "Mid_Age"
    return "Old"


def bucket_hr(value: float) -> str:
    if value < 45:
        return "Low"
    if value <= 100:
        return "Normal"
    return "High"


def bucket_rr(value: float) -> str:
    if value < 12:
        return "Low"
    if value <= 25:
        return "Normal"
    return "High"


def bucket_bt(value: float) -> str:
    if value < 36.4:
        return "Low"
    if value <= 37.6:
        return "Normal"
    return "High"


def _validate_columns(df: pd.DataFrame) -> None:
    required = set(RAW_FEATURE_COLUMNS + [RAW_TARGET_COLUMN])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")


def _map_numeric_column(df: pd.DataFrame, source: str, mapping: dict[int, str], target: str) -> None:
    numeric_values = pd.to_numeric(df[source], errors="coerce")
    df[target] = numeric_values.map(mapping)


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def preprocess_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw triage data into model-ready frame."""
    _validate_columns(df)
    frame = df[RAW_FEATURE_COLUMNS + [RAW_TARGET_COLUMN]].copy()
    frame.dropna(subset=RAW_FEATURE_COLUMNS + [RAW_TARGET_COLUMN], inplace=True)

    _map_numeric_column(frame, "Sex", SEX_MAP, "sex")
    _map_numeric_column(frame, "Injury", INJURY_MAP, "injury")
    _map_numeric_column(frame, "Pain", PAIN_MAP, "pain")
    _map_numeric_column(frame, "Mental", MENTAL_MAP, "mental_status")
    _map_numeric_column(frame, "Arrival mode", ARRIVAL_MODE_MAP, "arrival_mode")

    _coerce_numeric_columns(frame, ["Age", "HR", "RR", "BT", RAW_TARGET_COLUMN])
    frame = frame.dropna(subset=["Age", "HR", "RR", "BT", RAW_TARGET_COLUMN])

    frame = frame[
        (frame["Age"] >= 0)
        & (frame["Age"] <= 120)
        & (frame["HR"] > 0)
        & (frame["RR"] > 0)
        & (frame["BT"] > 0)
    ].copy()

    frame["age_group"] = frame["Age"].apply(bucket_age)
    frame["hr_group"] = frame["HR"].apply(bucket_hr)
    frame["rr_group"] = frame["RR"].apply(bucket_rr)
    frame["bt_group"] = frame["BT"].apply(bucket_bt)

    frame["chief_complain"] = frame["Chief_complain"].astype(str).str.strip()
    frame[MODEL_TARGET_COLUMN] = frame[RAW_TARGET_COLUMN].map(KTAS_TO_TRIAGE_MAP)

    frame = frame.dropna(
        subset=MODEL_FEATURE_COLUMNS + [MODEL_TARGET_COLUMN],
    )
    frame = frame[frame["chief_complain"].str.len() > 0].copy()

    # Deduplicate exact repeated records to avoid optimistic metrics from duplicates.
    frame = frame.drop_duplicates(subset=MODEL_FEATURE_COLUMNS + [MODEL_TARGET_COLUMN]).reset_index(drop=True)

    return frame[MODEL_FEATURE_COLUMNS + [MODEL_TARGET_COLUMN]]


def load_and_preprocess(csv_path: str | Path, *, read_csv_kwargs: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read CSV and return clean model frame."""
    kwargs = {"encoding": "ISO-8859-1", "sep": ";"}
    if read_csv_kwargs:
        kwargs.update(read_csv_kwargs)

    frame = pd.read_csv(Path(csv_path), **kwargs)
    return preprocess_frame(frame)

