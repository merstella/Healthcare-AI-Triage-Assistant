"""Deterministic triage agent: model decision + RAG evidence."""

from __future__ import annotations

from typing import Any, Mapping

from .inference import predict_from_features, predict_label_from_features
from .rag import retrieve_chunks


def run_triage_agent(
    *,
    model_bundle: Mapping[str, Any],
    kb_index_bundle: Mapping[str, Any],
    patient_features: Mapping[str, Any],
    user_question: str,
    top_k: int = 3,
) -> dict[str, Any]:
    probabilities = predict_from_features(model_bundle, patient_features)
    recommendation, applied_threshold = predict_label_from_features(model_bundle, patient_features)

    question = user_question.strip() if user_question.strip() else patient_features["chief_complain"]
    evidence = retrieve_chunks(dict(kb_index_bundle), question, top_k=top_k)

    evidence_sources = [item["source"] for item in evidence]
    if evidence:
        evidence_note = "Evidence retrieved from knowledge base sources listed below."
    else:
        evidence_note = "No matching KB chunk found; fallback to model-only recommendation."

    answer = (
        f"Recommended next step: {recommendation}\n"
        f"Reasoning: model threshold policy applied"
        f"{f' (emergency_threshold={applied_threshold:.3f})' if applied_threshold is not None else ''}. "
        f"{evidence_note}\n"
        "Safety note: This is a demo assistant and does not replace clinician judgment."
    )

    return {
        "recommendation": recommendation,
        "probabilities": probabilities,
        "applied_emergency_threshold": applied_threshold,
        "user_question": question,
        "evidence": evidence,
        "evidence_sources": evidence_sources,
        "answer": answer,
    }
