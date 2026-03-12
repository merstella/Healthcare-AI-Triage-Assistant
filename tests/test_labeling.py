from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_ai_agent.constants import TRIAGE_LABELS
from healthcare_ai_agent.labeling import decode_indices, encode_labels


class LabelEncodingTests(unittest.TestCase):
    def test_round_trip_encoding(self) -> None:
        encoded = encode_labels(TRIAGE_LABELS)
        decoded = decode_indices(encoded)
        self.assertEqual(decoded, TRIAGE_LABELS)

    def test_encoding_is_stable(self) -> None:
        encoded_once = encode_labels(["Emergency", "Doctor", "Self-care"])
        encoded_twice = encode_labels(["Emergency", "Doctor", "Self-care"])
        self.assertEqual(encoded_once, encoded_twice)
        self.assertEqual(encoded_once, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

