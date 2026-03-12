from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_ai_agent.recommendation import make_recommendation


class RecommendationPolicyTests(unittest.TestCase):
    def test_emergency_when_threshold_crossed(self) -> None:
        result = make_recommendation({"Emergency": 0.6, "Doctor": 0.3, "Self-care": 0.1})
        self.assertEqual(result.level, "Emergency")

    def test_self_care_allowed_only_when_emergency_is_low(self) -> None:
        result = make_recommendation({"Emergency": 0.1, "Doctor": 0.1, "Self-care": 0.8})
        self.assertEqual(result.level, "Self-care")

    def test_doctor_fallback_when_confidence_mixed(self) -> None:
        result = make_recommendation({"Emergency": 0.25, "Doctor": 0.3, "Self-care": 0.45})
        self.assertEqual(result.level, "Doctor")

    def test_missing_labels_default_to_zero(self) -> None:
        result = make_recommendation({"Emergency": 1.0})
        self.assertIn("Doctor", result.probabilities)
        self.assertIn("Self-care", result.probabilities)
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()

