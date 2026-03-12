"""Lightweight offline RAG utilities for recruiter demo."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _chunk_text(text: str, *, chunk_chars: int, overlap_chars: int) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if len(cleaned) <= chunk_chars:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_chars - overlap_chars)
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_chars)
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start += step
    return chunks


def build_kb_index(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    chunk_chars: int = 900,
    overlap_chars: int = 150,
) -> dict[str, Any]:
    source_dir = Path(input_dir)
    files = sorted(
        [
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}
        ]
    )
    if not files:
        raise ValueError(f"No .md/.txt files found under: {source_dir}")

    chunk_records: list[dict[str, Any]] = []
    for source_file in files:
        text = source_file.read_text(encoding="utf-8")
        for idx, chunk in enumerate(
            _chunk_text(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        ):
            chunk_records.append(
                {
                    "source": str(source_file),
                    "chunk_id": idx,
                    "text": chunk,
                }
            )

    if not chunk_records:
        raise ValueError("No text chunks created from KB files.")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([item["text"] for item in chunk_records])

    bundle: dict[str, Any] = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "records": chunk_records,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return bundle


def load_kb_index(index_path: str | Path) -> dict[str, Any]:
    bundle = joblib.load(Path(index_path))
    required = {"vectorizer", "matrix", "records"}
    missing = sorted(required.difference(bundle.keys()))
    if missing:
        raise ValueError(f"Invalid KB index bundle, missing keys: {missing}")
    return bundle


def retrieve_chunks(
    bundle: dict[str, Any],
    query: str,
    *,
    top_k: int = 3,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    vectorizer: TfidfVectorizer = bundle["vectorizer"]
    matrix = bundle["matrix"]
    records: list[dict[str, Any]] = bundle["records"]

    query_vec = vectorizer.transform([query])
    scores = (matrix @ query_vec.T).toarray().ravel()
    if scores.size == 0:
        return []

    ranked_indices = np.argsort(-scores)
    results: list[dict[str, Any]] = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score <= min_score:
            continue
        record = records[int(idx)]
        results.append(
            {
                "score": score,
                "source": record["source"],
                "chunk_id": int(record["chunk_id"]),
                "text": record["text"],
            }
        )
        if len(results) >= top_k:
            break
    return results
