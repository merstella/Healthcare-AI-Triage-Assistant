from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_ai_agent.constants import MODEL_FEATURE_COLUMNS, TRIAGE_LABELS
from healthcare_ai_agent.inference import (
    get_emergency_threshold_from_bundle,
    predict_label_from_features,
)
from healthcare_ai_agent.labeling import decode_index
from healthcare_ai_agent.modeling import build_pipeline, predict_with_emergency_policy


def _toy_training_frame() -> tuple[pd.DataFrame, list[int]]:
    rows = [
        ("Female", "Old", "Public Ambulance", "Yes", "chest pain with dyspnea", "Pain Response", "Yes", "High", "High", "High", 0),
        ("Male", "Mid_Age", "Public Ambulance", "Yes", "severe abdominal pain", "Verbose Response", "Yes", "High", "High", "High", 0),
        ("Male", "Adult", "Private Vehicle", "No", "persistent cough", "Alert", "Yes", "Normal", "Normal", "Normal", 1),
        ("Female", "Adult", "Walking", "No", "back pain", "Alert", "Yes", "Normal", "Normal", "Normal", 1),
        ("Female", "Young", "Walking", "No", "mild sore throat", "Alert", "No", "Normal", "Normal", "Normal", 2),
        ("Male", "Young", "Walking", "No", "runny nose", "Alert", "No", "Low", "Normal", "Low", 2),
    ]
    frame = pd.DataFrame(
        [
            {
                "sex": sex,
                "age_group": age_group,
                "arrival_mode": arrival_mode,
                "injury": injury,
                "chief_complain": chief_complain,
                "mental_status": mental_status,
                "pain": pain,
                "hr_group": hr_group,
                "rr_group": rr_group,
                "bt_group": bt_group,
            }
            for sex, age_group, arrival_mode, injury, chief_complain, mental_status, pain, hr_group, rr_group, bt_group, _ in rows
        ]
    )
    y = [target for *_, target in rows]
    return frame[MODEL_FEATURE_COLUMNS], y


class InferencePolicyTests(unittest.TestCase):
    def test_threshold_extracted_from_bundle_metadata(self) -> None:
        bundle = {"metadata": {"decision_policy": {"emergency_threshold": 0.155}}}
        self.assertAlmostEqual(get_emergency_threshold_from_bundle(bundle), 0.155)
        self.assertIsNone(get_emergency_threshold_from_bundle({}))

    def test_label_prediction_uses_tuned_threshold_policy(self) -> None:
        x, y = _toy_training_frame()
        model = build_pipeline(random_state=7)
        model.fit(x, y)

        threshold = 0.2
        bundle = {
            "model": model,
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "labels": TRIAGE_LABELS,
            "metadata": {"decision_policy": {"emergency_threshold": threshold}},
        }
        features = x.iloc[0].to_dict()

        label, applied_threshold = predict_label_from_features(bundle, features)
        expected_idx = predict_with_emergency_policy(
            model,
            pd.DataFrame([features], columns=MODEL_FEATURE_COLUMNS),
            emergency_threshold=threshold,
        )[0]

        self.assertEqual(label, decode_index(expected_idx))
        self.assertAlmostEqual(applied_threshold, threshold)


if __name__ == "__main__":
    unittest.main()
