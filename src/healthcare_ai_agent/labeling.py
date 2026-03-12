"""Stable label encoding helpers."""

from __future__ import annotations

from typing import Iterable

from .constants import TRIAGE_LABELS


LABEL_TO_INDEX = {label: idx for idx, label in enumerate(TRIAGE_LABELS)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}


def encode_label(label: str) -> int:
    """Convert triage label text to stable index."""
    if label not in LABEL_TO_INDEX:
        raise ValueError(f"Unknown triage label: {label}")
    return LABEL_TO_INDEX[label]


def decode_index(index: int) -> str:
    """Convert stable index back to triage label text."""
    if index not in INDEX_TO_LABEL:
        raise ValueError(f"Unknown triage index: {index}")
    return INDEX_TO_LABEL[index]


def encode_labels(labels: Iterable[str]) -> list[int]:
    return [encode_label(label) for label in labels]


def decode_indices(indices: Iterable[int]) -> list[str]:
    return [decode_index(idx) for idx in indices]

