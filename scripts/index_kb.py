#!/usr/bin/env python3
"""Build offline KB index for RAG demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_ai_agent.rag import build_kb_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KB index for offline RAG.")
    parser.add_argument("--kb-dir", default="kb/docs", help="Directory containing .md/.txt files.")
    parser.add_argument("--output-path", default="artifacts/kb_index.joblib", help="Index artifact path.")
    parser.add_argument("--chunk-chars", type=int, default=900, help="Chunk size by character count.")
    parser.add_argument("--overlap-chars", type=int, default=150, help="Chunk overlap by character count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build_kb_index(
        args.kb_dir,
        args.output_path,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )
    print(f"Saved KB index to: {args.output_path}")
    print(f"Chunks indexed: {len(bundle['records'])}")


if __name__ == "__main__":
    main()
