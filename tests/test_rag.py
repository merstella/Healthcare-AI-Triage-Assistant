from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_ai_agent.rag import build_kb_index, load_kb_index, retrieve_chunks


class RagTests(unittest.TestCase):
    def test_build_and_retrieve_kb_index(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            kb_dir = Path(tmp_dir) / "docs"
            kb_dir.mkdir(parents=True, exist_ok=True)
            (kb_dir / "chest.md").write_text(
                "Chest pain with breathing difficulty is a high risk symptom.",
                encoding="utf-8",
            )
            (kb_dir / "minor.md").write_text(
                "Minor sore throat usually can be monitored.",
                encoding="utf-8",
            )

            index_path = Path(tmp_dir) / "kb_index.joblib"
            build_kb_index(kb_dir, index_path)
            bundle = load_kb_index(index_path)

            results = retrieve_chunks(bundle, "chest pain emergency", top_k=1)
            self.assertEqual(len(results), 1)
            self.assertIn("chest.md", results[0]["source"])
            self.assertGreater(results[0]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
