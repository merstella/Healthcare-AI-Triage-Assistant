#!/usr/bin/env python3
"""Advanced benchmark + training pipeline with CV, embeddings, calibration, and slice metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

# Avoid matplotlib cache warning when importing catboost stack in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_ai_agent.constants import CATEGORICAL_FEATURE_COLUMNS, TRIAGE_LABELS
from healthcare_ai_agent.labeling import encode_label
from healthcare_ai_agent.modeling import split_xy
from healthcare_ai_agent.preprocessing import load_and_preprocess


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    model_type: str
    text_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced CV + holdout benchmark for triage models.")
    parser.add_argument("--data-path", required=True, help="Path to source CSV.")
    parser.add_argument("--output-path", default="artifacts/advanced_metrics.json", help="Output report JSON path.")
    parser.add_argument(
        "--model-output-path",
        default="artifacts/advanced_model_bundle.joblib",
        help="Output path for selected advanced model bundle.",
    )
    parser.add_argument("--holdout-size", type=float, default=0.2, help="Holdout size ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--cv-folds", type=int, default=5, help="StratifiedKFold fold count.")
    parser.add_argument("--emergency-recall-target", type=float, default=0.95, help="Recall constraint for Emergency.")
    parser.add_argument("--threshold-grid-size", type=int, default=401, help="Threshold search grid size.")
    parser.add_argument("--fn-cost", type=float, default=10.0, help="False negative cost for Emergency.")
    parser.add_argument("--fp-cost", type=float, default=1.0, help="False positive cost for Emergency.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence embedding model id.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=64, help="Embedding batch size.")
    return parser.parse_args()


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
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


def medical_cost(
    y_true: list[int],
    y_pred: list[int],
    *,
    emergency_class: int,
    fn_cost: float,
    fp_cost: float,
) -> float:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    false_negative = int(np.sum((y_true_arr == emergency_class) & (y_pred_arr != emergency_class)))
    false_positive = int(np.sum((y_true_arr != emergency_class) & (y_pred_arr == emergency_class)))
    return float(false_negative * fn_cost + false_positive * fp_cost)


def predict_with_emergency_threshold(
    probabilities: np.ndarray,
    class_indices: list[int],
    *,
    threshold: float | None,
) -> list[int]:
    if threshold is None:
        return [class_indices[int(np.argmax(row))] for row in probabilities]

    emergency_class = encode_label("Emergency")
    class_to_col = {class_idx: col_idx for col_idx, class_idx in enumerate(class_indices)}
    if emergency_class not in class_to_col:
        return [class_indices[int(np.argmax(row))] for row in probabilities]

    emergency_col = class_to_col[emergency_class]
    non_emergency_classes = [idx for idx in class_indices if idx != emergency_class]
    predictions: list[int] = []
    for row in probabilities:
        if float(row[emergency_col]) >= threshold:
            predictions.append(emergency_class)
        else:
            best_non_emergency = max(non_emergency_classes, key=lambda idx: float(row[class_to_col[idx]]))
            predictions.append(int(best_non_emergency))
    return predictions


def tune_threshold(
    probabilities: np.ndarray,
    class_indices: list[int],
    y_true: list[int],
    *,
    recall_target: float,
    grid_size: int,
    fn_cost: float,
    fp_cost: float,
) -> dict[str, Any]:
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")

    emergency_class = encode_label("Emergency")
    all_candidates: list[dict[str, Any]] = []
    for threshold in np.linspace(0.0, 1.0, grid_size):
        threshold_value = float(threshold)
        y_pred = predict_with_emergency_threshold(
            probabilities,
            class_indices,
            threshold=threshold_value,
        )
        metrics = compute_metrics(y_true, y_pred)
        cost = medical_cost(
            y_true,
            y_pred,
            emergency_class=emergency_class,
            fn_cost=fn_cost,
            fp_cost=fp_cost,
        )
        all_candidates.append(
            {
                "threshold": threshold_value,
                "metrics": metrics,
                "macro_f1": metrics["macro_f1"],
                "emergency_recall": metrics["emergency_recall"],
                "medical_cost": cost,
            }
        )

    valid = [item for item in all_candidates if item["emergency_recall"] >= recall_target]
    if valid:
        best = max(valid, key=lambda item: (item["macro_f1"], -item["medical_cost"], item["threshold"]))
        constraint_met = True
    else:
        best = max(
            all_candidates,
            key=lambda item: (item["emergency_recall"], item["macro_f1"], -item["medical_cost"]),
        )
        constraint_met = False

    return {
        "best_threshold": best["threshold"],
        "best_macro_f1": best["macro_f1"],
        "best_emergency_recall": best["emergency_recall"],
        "best_medical_cost": best["medical_cost"],
        "constraint_met": constraint_met,
        "target_recall": recall_target,
        "searched_threshold_count": len(all_candidates),
        "best_metrics": best["metrics"],
    }


def build_estimator(
    model_type: str,
    *,
    class_weights: dict[int, float],
    random_state: int,
):
    if model_type == "logreg":
        return LogisticRegression(
            max_iter=4000,
            class_weight=class_weights,
            random_state=random_state,
        )
    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="multiclass",
            num_class=len(TRIAGE_LABELS),
            class_weight=class_weights,
            n_estimators=220,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )
    if model_type == "catboost":
        from catboost import CatBoostClassifier

        ordered_weights = [class_weights[idx] for idx in range(len(TRIAGE_LABELS))]
        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=180,
            learning_rate=0.05,
            depth=6,
            class_weights=ordered_weights,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def fit_feature_encoders(
    x_train_fold,
    *,
    text_mode: str,
):
    one_hot = OneHotEncoder(handle_unknown="ignore")
    cat_train = one_hot.fit_transform(x_train_fold[CATEGORICAL_FEATURE_COLUMNS])
    if text_mode == "tfidf":
        tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
        text_train = tfidf.fit_transform(x_train_fold["chief_complain"].astype(str).tolist())
        return one_hot, tfidf, hstack([cat_train, text_train], format="csr")
    if text_mode == "embedding":
        return one_hot, None, cat_train
    raise ValueError(f"Unsupported text_mode: {text_mode}")


def transform_features(
    x_fold,
    *,
    text_mode: str,
    one_hot: OneHotEncoder,
    tfidf: TfidfVectorizer | None,
    embeddings: np.ndarray | None,
):
    cat_part = one_hot.transform(x_fold[CATEGORICAL_FEATURE_COLUMNS])
    if text_mode == "tfidf":
        assert tfidf is not None
        text_part = tfidf.transform(x_fold["chief_complain"].astype(str).tolist())
        return hstack([cat_part, text_part], format="csr")
    if text_mode == "embedding":
        assert embeddings is not None
        return hstack([cat_part, csr_matrix(embeddings)], format="csr")
    raise ValueError(f"Unsupported text_mode: {text_mode}")


def maybe_dense(matrix, *, model_type: str):
    if model_type == "catboost" and issparse(matrix):
        return matrix.toarray()
    return matrix


def class_weight_dict(y_train_fold: list[int]) -> dict[int, float]:
    classes = np.array(list(range(len(TRIAGE_LABELS))))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(y_train_fold),
    )
    return {int(class_idx): float(weight) for class_idx, weight in zip(classes, weights)}


def oof_probabilities_for_candidate(
    x_train,
    y_train: list[int],
    *,
    candidate: CandidateSpec,
    calibration_method: str,
    cv_folds: int,
    random_state: int,
    train_embeddings: np.ndarray | None,
) -> tuple[np.ndarray, list[int]]:
    class_indices = list(range(len(TRIAGE_LABELS)))
    oof_probs = np.zeros((len(y_train), len(class_indices)), dtype=float)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    y_arr = np.array(y_train)

    for fold_id, (train_idx, val_idx) in enumerate(splitter.split(x_train, y_arr), start=1):
        x_fit = x_train.iloc[train_idx]
        x_val = x_train.iloc[val_idx]
        y_fit = [y_train[i] for i in train_idx]

        one_hot, tfidf, x_fit_features_base = fit_feature_encoders(
            x_fit,
            text_mode=candidate.text_mode,
        )
        x_val_embeddings = None
        if candidate.text_mode == "embedding":
            assert train_embeddings is not None
            x_fit_embeddings = train_embeddings[train_idx]
            x_val_embeddings = train_embeddings[val_idx]
            x_fit_features = hstack([x_fit_features_base, csr_matrix(x_fit_embeddings)], format="csr")
        else:
            x_fit_features = x_fit_features_base

        x_val_features = transform_features(
            x_val,
            text_mode=candidate.text_mode,
            one_hot=one_hot,
            tfidf=tfidf,
            embeddings=x_val_embeddings,
        )

        weights = class_weight_dict(y_fit)
        estimator = build_estimator(
            candidate.model_type,
            class_weights=weights,
            random_state=random_state + fold_id,
        )

        x_fit_model = maybe_dense(x_fit_features, model_type=candidate.model_type)
        x_val_model = maybe_dense(x_val_features, model_type=candidate.model_type)

        if calibration_method == "none":
            estimator.fit(x_fit_model, y_fit)
            fold_probs = estimator.predict_proba(x_val_model)
            fold_classes = [int(item) for item in estimator.classes_]
        else:
            calibrated = CalibratedClassifierCV(
                estimator=estimator,
                method=calibration_method,
                cv=2,
            )
            calibrated.fit(x_fit_model, y_fit)
            fold_probs = calibrated.predict_proba(x_val_model)
            fold_classes = [int(item) for item in calibrated.classes_]

        class_to_col = {class_idx: col_idx for col_idx, class_idx in enumerate(fold_classes)}
        for global_class_idx in class_indices:
            if global_class_idx not in class_to_col:
                continue
            oof_probs[val_idx, global_class_idx] = fold_probs[:, class_to_col[global_class_idx]]

    return oof_probs, class_indices


def score_entry(entry: dict[str, Any]) -> tuple[int, float, float, float]:
    meets_constraint = 1 if entry["threshold_tuning"]["constraint_met"] else 0
    emergency_recall = float(entry["cv_metrics"]["emergency_recall"])
    macro_f1 = float(entry["cv_metrics"]["macro_f1"])
    negative_cost = -float(entry["threshold_tuning"]["best_medical_cost"])
    if meets_constraint:
        return (1, macro_f1, negative_cost, emergency_recall)
    return (0, emergency_recall, macro_f1, negative_cost)


def evaluate_slices(
    y_true: list[int],
    y_pred: list[int],
    slice_values,
) -> list[dict[str, Any]]:
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    results: list[dict[str, Any]] = []
    for value in sorted(slice_values.dropna().unique().tolist()):
        mask = slice_values == value
        idx = np.where(mask.to_numpy())[0]
        if len(idx) == 0:
            continue
        part_true = y_true_arr[idx].tolist()
        part_pred = y_pred_arr[idx].tolist()
        metrics = compute_metrics(part_true, part_pred)
        results.append(
            {
                "slice_value": value,
                "support": len(idx),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "emergency_recall": metrics["emergency_recall"],
            }
        )
    return results


def main() -> None:
    args = parse_args()
    frame = load_and_preprocess(args.data_path)
    x, y = split_xy(frame)

    x_train_raw, x_holdout_raw, y_train_raw, y_holdout_raw = train_test_split(
        x,
        y,
        test_size=args.holdout_size,
        random_state=args.random_state,
        stratify=y,
    )

    train_indices = x_train_raw.index.to_numpy()
    holdout_indices = x_holdout_raw.index.to_numpy()
    x_train = x_train_raw.reset_index(drop=True)
    x_holdout = x_holdout_raw.reset_index(drop=True)
    y_train = list(y_train_raw)
    y_holdout = list(y_holdout_raw)

    has_lightgbm = True
    has_catboost = True
    has_sentence_embeddings = True
    try:
        import lightgbm  # noqa: F401
    except Exception:
        has_lightgbm = False
    try:
        import catboost  # noqa: F401
    except Exception:
        has_catboost = False
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception:
        has_sentence_embeddings = False

    family_specs: list[CandidateSpec] = [CandidateSpec("logreg_tfidf", "logreg", "tfidf")]
    if has_sentence_embeddings:
        family_specs.append(CandidateSpec("logreg_embedding", "logreg", "embedding"))
        if has_lightgbm:
            family_specs.append(CandidateSpec("lightgbm_embedding", "lightgbm", "embedding"))
        if has_catboost:
            family_specs.append(CandidateSpec("catboost_embedding", "catboost", "embedding"))

    if not family_specs:
        raise RuntimeError("No model family available to run benchmark.")

    train_embeddings = None
    holdout_embeddings = None
    if any(spec.text_mode == "embedding" for spec in family_specs):
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(args.embedding_model)
        all_text = x["chief_complain"].astype(str).tolist()
        all_embeddings = np.asarray(
            embedder.encode(
                all_text,
                batch_size=args.embedding_batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        train_embeddings = all_embeddings[train_indices]
        holdout_embeddings = all_embeddings[holdout_indices]

    family_results: list[dict[str, Any]] = []
    for spec in family_specs:
        print(f"[CV family] {spec.name} ...", flush=True)
        oof_probs, class_indices = oof_probabilities_for_candidate(
            x_train,
            y_train,
            candidate=spec,
            calibration_method="none",
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            train_embeddings=train_embeddings,
        )
        threshold_result = tune_threshold(
            oof_probs,
            class_indices,
            y_train,
            recall_target=args.emergency_recall_target,
            grid_size=args.threshold_grid_size,
            fn_cost=args.fn_cost,
            fp_cost=args.fp_cost,
        )
        y_pred_cv = predict_with_emergency_threshold(
            oof_probs,
            class_indices,
            threshold=threshold_result["best_threshold"],
        )
        cv_metrics = compute_metrics(y_train, y_pred_cv)
        family_results.append(
            {
                "family": spec.name,
                "model_type": spec.model_type,
                "text_mode": spec.text_mode,
                "calibration_method": "none",
                "cv_metrics": cv_metrics,
                "threshold_tuning": threshold_result,
            }
        )

    best_family = max(family_results, key=score_entry)
    selected_spec = next(spec for spec in family_specs if spec.name == best_family["family"])

    calibration_results: list[dict[str, Any]] = []
    for calibration_method in ["none", "sigmoid", "isotonic"]:
        print(f"[CV calibration] {selected_spec.name} + {calibration_method} ...", flush=True)
        oof_probs, class_indices = oof_probabilities_for_candidate(
            x_train,
            y_train,
            candidate=selected_spec,
            calibration_method=calibration_method,
            cv_folds=args.cv_folds,
            random_state=args.random_state,
            train_embeddings=train_embeddings,
        )
        threshold_result = tune_threshold(
            oof_probs,
            class_indices,
            y_train,
            recall_target=args.emergency_recall_target,
            grid_size=args.threshold_grid_size,
            fn_cost=args.fn_cost,
            fp_cost=args.fp_cost,
        )
        y_pred_cv = predict_with_emergency_threshold(
            oof_probs,
            class_indices,
            threshold=threshold_result["best_threshold"],
        )
        cv_metrics = compute_metrics(y_train, y_pred_cv)
        calibration_results.append(
            {
                "family": selected_spec.name,
                "model_type": selected_spec.model_type,
                "text_mode": selected_spec.text_mode,
                "calibration_method": calibration_method,
                "cv_metrics": cv_metrics,
                "threshold_tuning": threshold_result,
            }
        )

    best_config = max(calibration_results, key=score_entry)
    chosen_threshold = float(best_config["threshold_tuning"]["best_threshold"])

    # Final fit on full train split.
    one_hot, tfidf, x_train_base = fit_feature_encoders(
        x_train,
        text_mode=selected_spec.text_mode,
    )
    holdout_embed = None
    if selected_spec.text_mode == "embedding":
        assert train_embeddings is not None and holdout_embeddings is not None
        x_train_features = hstack([x_train_base, csr_matrix(train_embeddings)], format="csr")
        holdout_embed = holdout_embeddings
    else:
        x_train_features = x_train_base

    x_holdout_features = transform_features(
        x_holdout,
        text_mode=selected_spec.text_mode,
        one_hot=one_hot,
        tfidf=tfidf,
        embeddings=holdout_embed,
    )

    full_weights = class_weight_dict(y_train)
    final_estimator = build_estimator(
        selected_spec.model_type,
        class_weights=full_weights,
        random_state=args.random_state,
    )
    x_train_model = maybe_dense(x_train_features, model_type=selected_spec.model_type)
    x_holdout_model = maybe_dense(x_holdout_features, model_type=selected_spec.model_type)

    if best_config["calibration_method"] == "none":
        final_estimator.fit(x_train_model, y_train)
        holdout_probs = final_estimator.predict_proba(x_holdout_model)
        holdout_classes = [int(item) for item in final_estimator.classes_]
        deployed_model = final_estimator
    else:
        final_calibrated = CalibratedClassifierCV(
            estimator=final_estimator,
            method=best_config["calibration_method"],
            cv=3,
        )
        final_calibrated.fit(x_train_model, y_train)
        holdout_probs = final_calibrated.predict_proba(x_holdout_model)
        holdout_classes = [int(item) for item in final_calibrated.classes_]
        deployed_model = final_calibrated

    y_holdout_pred = predict_with_emergency_threshold(
        holdout_probs,
        holdout_classes,
        threshold=chosen_threshold,
    )
    holdout_metrics = compute_metrics(y_holdout, y_holdout_pred)
    holdout_cost = medical_cost(
        y_holdout,
        y_holdout_pred,
        emergency_class=encode_label("Emergency"),
        fn_cost=args.fn_cost,
        fp_cost=args.fp_cost,
    )

    slice_metrics = {
        "sex": evaluate_slices(y_holdout, y_holdout_pred, x_holdout["sex"]),
        "age_group": evaluate_slices(y_holdout, y_holdout_pred, x_holdout["age_group"]),
    }

    report = {
        "config": {
            "data_path": str(Path(args.data_path).resolve()),
            "model_output_path": str(Path(args.model_output_path).resolve()),
            "holdout_size": args.holdout_size,
            "random_state": args.random_state,
            "cv_folds": args.cv_folds,
            "emergency_recall_target": args.emergency_recall_target,
            "threshold_grid_size": args.threshold_grid_size,
            "fn_cost": args.fn_cost,
            "fp_cost": args.fp_cost,
            "embedding_model": args.embedding_model,
            "embedding_batch_size": args.embedding_batch_size,
        },
        "dataset": {
            "total_rows_after_preprocessing": len(frame),
            "train_rows": len(x_train),
            "holdout_rows": len(x_holdout),
        },
        "model_family_comparison_cv": family_results,
        "selected_family": best_family,
        "calibration_comparison_cv": calibration_results,
        "selected_configuration": best_config,
        "holdout_metrics": holdout_metrics,
        "holdout_medical_cost": holdout_cost,
        "applied_emergency_threshold": chosen_threshold,
        "slice_metrics": slice_metrics,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    model_output_path = Path(args.model_output_path)
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    model_bundle = {
        "model": deployed_model,
        "one_hot_encoder": one_hot,
        "tfidf_vectorizer": tfidf,
        "text_mode": selected_spec.text_mode,
        "model_type": selected_spec.model_type,
        "calibration_method": best_config["calibration_method"],
        "class_indices": holdout_classes,
        "emergency_threshold": chosen_threshold,
        "categorical_feature_columns": CATEGORICAL_FEATURE_COLUMNS,
        "text_feature_column": "chief_complain",
        "embedding_model": (
            args.embedding_model if selected_spec.text_mode == "embedding" else None
        ),
        "embedding_normalize": bool(selected_spec.text_mode == "embedding"),
    }
    joblib.dump(model_bundle, model_output_path)

    print(f"Saved advanced report to: {output_path}")
    print(f"Saved advanced model bundle to: {model_output_path}")
    print(
        "Final holdout -> "
        f"macro_f1={holdout_metrics['macro_f1']:.4f}, "
        f"emergency_recall={holdout_metrics['emergency_recall']:.4f}, "
        f"medical_cost={holdout_cost:.2f}, "
        f"threshold={chosen_threshold:.3f}"
    )
    print(
        "Selected config -> "
        f"family={best_config['family']}, "
        f"calibration={best_config['calibration_method']}"
    )


if __name__ == "__main__":
    main()
