from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_ai_agent.agent import run_triage_agent
from healthcare_ai_agent.constants import MODEL_FEATURE_COLUMNS, TRIAGE_LABELS
from healthcare_ai_agent.modeling import build_pipeline
from healthcare_ai_agent.rag import build_kb_index, load_kb_index


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


class AgentTests(unittest.TestCase):
    def test_agent_returns_recommendation_and_evidence(self) -> None:
        x, y = _toy_training_frame()
        model = build_pipeline(random_state=11)
        model.fit(x, y)
        model_bundle = {
            "model": model,
            "feature_columns": MODEL_FEATURE_COLUMNS,
            "labels": TRIAGE_LABELS,
            "metadata": {"decision_policy": {"emergency_threshold": 0.2}},
        }

        with TemporaryDirectory() as tmp_dir:
            kb_dir = Path(tmp_dir) / "docs"
            kb_dir.mkdir(parents=True, exist_ok=True)
            (kb_dir / "red_flags.md").write_text(
                "Chest pain with dyspnea should be urgently assessed in emergency care.",
                encoding="utf-8",
            )
            index_path = Path(tmp_dir) / "kb_index.joblib"
            build_kb_index(kb_dir, index_path)
            kb_bundle = load_kb_index(index_path)

            result = run_triage_agent(
                model_bundle=model_bundle,
                kb_index_bundle=kb_bundle,
                patient_features=x.iloc[0].to_dict(),
                user_question="What to do with chest pain and shortness of breath?",
                top_k=2,
            )

            self.assertIn(result["recommendation"], TRIAGE_LABELS)
            self.assertIn("Emergency", result["probabilities"])
            self.assertGreaterEqual(len(result["evidence"]), 1)


if __name__ == "__main__":
    unittest.main()
