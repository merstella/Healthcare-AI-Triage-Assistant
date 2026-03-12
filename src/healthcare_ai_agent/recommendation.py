"""Safety-first recommendation policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .constants import TRIAGE_LABELS


@dataclass(frozen=True)
class RecommendationThresholds:
    emergency_threshold: float = 0.50
    emergency_watch_threshold: float = 0.35
    doctor_threshold: float = 0.45
    self_care_threshold: float = 0.60
    emergency_guardrail_for_self_care: float = 0.20


@dataclass(frozen=True)
class RecommendationResult:
    level: str
    rationale: str
    probabilities: dict[str, float]


DEFAULT_THRESHOLDS = RecommendationThresholds()


def _normalize_probabilities(probabilities: Mapping[str, float]) -> dict[str, float]:
    normalized = {label: float(probabilities.get(label, 0.0)) for label in TRIAGE_LABELS}
    total = sum(normalized.values())
    if total > 0:
        normalized = {label: value / total for label, value in normalized.items()}
    return normalized


def make_recommendation(
    probabilities: Mapping[str, float],
    *,
    thresholds: RecommendationThresholds = DEFAULT_THRESHOLDS,
) -> RecommendationResult:
    probs = _normalize_probabilities(probabilities)

    emergency = probs["Emergency"]
    doctor = probs["Doctor"]
    self_care = probs["Self-care"]
    top_label, top_prob = max(probs.items(), key=lambda item: item[1])

    if emergency >= thresholds.emergency_threshold:
        return RecommendationResult(
            level="Emergency",
            rationale="Emergency probability crosses hard safety threshold.",
            probabilities=probs,
        )

    if top_label == "Emergency" and emergency >= thresholds.emergency_watch_threshold:
        return RecommendationResult(
            level="Emergency",
            rationale="Emergency is the top class and above watch threshold.",
            probabilities=probs,
        )

    if doctor >= thresholds.doctor_threshold:
        return RecommendationResult(
            level="Doctor",
            rationale="Doctor probability crosses consultation threshold.",
            probabilities=probs,
        )

    if (
        self_care >= thresholds.self_care_threshold
        and emergency < thresholds.emergency_guardrail_for_self_care
    ):
        return RecommendationResult(
            level="Self-care",
            rationale="Self-care confidence is high while emergency risk is low.",
            probabilities=probs,
        )

    return RecommendationResult(
        level="Doctor",
        rationale="Fallback to doctor for safety when confidence is mixed.",
        probabilities=probs,
    )

