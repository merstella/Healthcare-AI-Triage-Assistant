"""Model training and evaluation logic."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .constants import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    MODEL_TARGET_COLUMN,
    TEXT_FEATURE_COLUMN,
    TRIAGE_LABELS,
)
from .labeling import decode_index, encode_label, encode_labels


def build_pipeline(*, random_state: int = 42) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURE_COLUMNS),
            (
                "chief_complain_tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000),
                TEXT_FEATURE_COLUMN,
            ),
        ],
    )
    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def split_xy(frame: Any) -> tuple[Any, list[int]]:
    x = frame[MODEL_FEATURE_COLUMNS].copy()
    y = encode_labels(frame[MODEL_TARGET_COLUMN].tolist())
    return x, y


def _evaluate_from_predictions(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    labels = list(range(len(TRIAGE_LABELS)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=TRIAGE_LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "emergency_recall": float(report["Emergency"]["recall"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": report,
    }


def predict_with_emergency_policy(
    model: Pipeline,
    x: Any,
    *,
    emergency_threshold: float | None = None,
) -> list[int]:
    if emergency_threshold is None:
        return model.predict(x).tolist()

    probabilities = model.predict_proba(x)
    class_indices = [int(item) for item in model.named_steps["classifier"].classes_.tolist()]
    return predict_with_emergency_policy_from_probabilities(
        probabilities,
        class_indices,
        emergency_threshold=emergency_threshold,
    )


def evaluate_model(
    model: Pipeline,
    x: Any,
    y_true: list[int],
    *,
    emergency_threshold: float | None = None,
) -> dict[str, Any]:
    y_pred = predict_with_emergency_policy(
        model,
        x,
        emergency_threshold=emergency_threshold,
    )
    return _evaluate_from_predictions(y_true, y_pred)


def tune_emergency_threshold(
    model: Pipeline,
    x_tune: Any,
    y_tune: list[int],
    *,
    recall_target: float = 0.95,
    grid_size: int = 201,
) -> dict[str, Any]:
    probabilities = model.predict_proba(x_tune)
    class_indices = [int(item) for item in model.named_steps["classifier"].classes_.tolist()]
    return tune_emergency_threshold_from_probabilities(
        probabilities,
        class_indices,
        y_tune,
        recall_target=recall_target,
        grid_size=grid_size,
    )


def predict_with_emergency_policy_from_probabilities(
    probabilities: np.ndarray,
    class_indices: list[int],
    *,
    emergency_threshold: float | None,
) -> list[int]:
    if emergency_threshold is None:
        return [class_indices[int(np.argmax(row))] for row in probabilities]

    emergency_class = encode_label("Emergency")
    class_to_column = {class_idx: col_idx for col_idx, class_idx in enumerate(class_indices)}
    if emergency_class not in class_to_column:
        return [class_indices[int(np.argmax(row))] for row in probabilities]

    emergency_col = class_to_column[emergency_class]
    non_emergency_classes = [idx for idx in class_indices if idx != emergency_class]
    if not non_emergency_classes:
        return [emergency_class for _ in range(len(probabilities))]

    predictions: list[int] = []
    for row in probabilities:
        emergency_probability = float(row[emergency_col])
        if emergency_probability >= emergency_threshold:
            predictions.append(emergency_class)
            continue
        best_non_emergency = max(
            non_emergency_classes,
            key=lambda class_idx: float(row[class_to_column[class_idx]]),
        )
        predictions.append(int(best_non_emergency))
    return predictions


def tune_emergency_threshold_from_probabilities(
    probabilities: np.ndarray,
    class_indices: list[int],
    y_true: list[int],
    *,
    recall_target: float = 0.95,
    grid_size: int = 201,
) -> dict[str, Any]:
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")

    thresholds = np.linspace(0.0, 1.0, num=grid_size)
    all_candidates: list[dict[str, Any]] = []

    for threshold in thresholds:
        threshold_value = float(threshold)
        y_pred = predict_with_emergency_policy_from_probabilities(
            probabilities,
            class_indices,
            emergency_threshold=threshold_value,
        )
        metrics = _evaluate_from_predictions(y_true, y_pred)
        all_candidates.append(
            {
                "threshold": threshold_value,
                "macro_f1": metrics["macro_f1"],
                "emergency_recall": metrics["emergency_recall"],
                "metrics": metrics,
            }
        )

    candidates_meeting_recall = [
        item for item in all_candidates if item["emergency_recall"] >= recall_target
    ]

    if candidates_meeting_recall:
        best = max(
            candidates_meeting_recall,
            key=lambda item: (item["macro_f1"], item["threshold"]),
        )
        constraint_met = True
    else:
        # If target is impossible with this model, maximize recall first, then macro-F1.
        best = max(
            all_candidates,
            key=lambda item: (item["emergency_recall"], item["macro_f1"], -item["threshold"]),
        )
        constraint_met = False

    return {
        "best_threshold": best["threshold"],
        "best_macro_f1": best["macro_f1"],
        "best_emergency_recall": best["emergency_recall"],
        "constraint_met": constraint_met,
        "target_recall": float(recall_target),
        "searched_threshold_count": len(all_candidates),
        "best_metrics_on_tuning_split": best["metrics"],
    }


def compute_oof_probabilities(
    base_model: Pipeline,
    x_train: Any,
    y_train: list[int],
    *,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, list[int]]:
    if cv_folds < 2:
        raise ValueError("cv_folds must be >= 2")

    class_indices = list(range(len(TRIAGE_LABELS)))
    oof_probabilities = np.zeros((len(y_train), len(class_indices)), dtype=float)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    y_train_array = np.array(y_train)
    for train_idx, val_idx in splitter.split(x_train, y_train_array):
        fold_model = clone(base_model)
        x_fit = x_train.iloc[train_idx]
        y_fit = [y_train[i] for i in train_idx]
        x_val = x_train.iloc[val_idx]

        fold_model.fit(x_fit, y_fit)
        fold_probabilities = fold_model.predict_proba(x_val)
        fold_classes = [int(item) for item in fold_model.named_steps["classifier"].classes_.tolist()]
        fold_class_to_col = {class_idx: col_idx for col_idx, class_idx in enumerate(fold_classes)}

        for global_class_idx in class_indices:
            if global_class_idx not in fold_class_to_col:
                continue
            oof_probabilities[val_idx, global_class_idx] = fold_probabilities[
                :,
                fold_class_to_col[global_class_idx],
            ]

    return oof_probabilities, class_indices


def train_model(
    frame: Any,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    emergency_recall_target: float = 0.95,
    threshold_grid_size: int = 201,
    threshold_tuning_cv_folds: int = 5,
) -> dict[str, Any]:
    x, y = split_xy(frame)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    base_model = build_pipeline(random_state=random_state)
    oof_probabilities, oof_class_indices = compute_oof_probabilities(
        base_model,
        x_train_full,
        y_train_full,
        cv_folds=threshold_tuning_cv_folds,
        random_state=random_state,
    )
    threshold_tuning = tune_emergency_threshold_from_probabilities(
        oof_probabilities,
        oof_class_indices,
        y_train_full,
        recall_target=emergency_recall_target,
        grid_size=threshold_grid_size,
    )
    best_threshold = threshold_tuning["best_threshold"]

    # Train final model on full training split after selecting threshold from OOF predictions.
    model = build_pipeline(random_state=random_state)
    model.fit(x_train_full, y_train_full)

    holdout_metrics = evaluate_model(
        model,
        x_test,
        y_test,
        emergency_threshold=best_threshold,
    )
    train_metrics = evaluate_model(
        model,
        x_train_full,
        y_train_full,
        emergency_threshold=best_threshold,
    )

    return {
        "model": model,
        "x_train_size": len(x_train_full),
        "x_test_size": len(x_test),
        "x_tuning_size": len(x_train_full),
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "threshold_tuning": threshold_tuning,
        "threshold_tuning_cv_folds": threshold_tuning_cv_folds,
    }


def predict_proba_dict(model: Pipeline, x_row: Any) -> dict[str, float]:
    probabilities = model.predict_proba(x_row)[0]
    class_indices = [int(item) for item in model.named_steps["classifier"].classes_.tolist()]
    result = {
        decode_index(class_idx): float(probability)
        for class_idx, probability in zip(class_indices, probabilities)
    }

    # Guarantee all classes appear in output dict.
    for label in TRIAGE_LABELS:
        result.setdefault(label, 0.0)
    return result
