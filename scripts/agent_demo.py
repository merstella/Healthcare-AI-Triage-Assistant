#!/usr/bin/env python3
"""Interactive agent demo: triage decision + KB retrieval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from healthcare_ai_agent.agent import run_triage_agent
from healthcare_ai_agent.cli import collect_patient_features
from healthcare_ai_agent.inference import load_model_bundle
from healthcare_ai_agent.rag import load_kb_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run triage + RAG agent demo.")
    parser.add_argument("--model-path", default="artifacts/triage_model.joblib", help="Model bundle path.")
    parser.add_argument("--kb-index-path", default="artifacts/kb_index.joblib", help="KB index bundle path.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of KB chunks to retrieve.")
    return parser.parse_args()


def _print_agent_result(result: dict) -> None:
    print("\nAutomated probabilities:")
    for label in ("Emergency", "Doctor", "Self-care"):
        print(f"- {label}: {result['probabilities'][label] * 100:.2f}%")

    print(f"\nRecommended next step: {result['recommendation']}")
    print(result["answer"])

    print("\nRetrieved evidence:")
    if not result["evidence"]:
        print("- No evidence chunk found for this query.")
    else:
        for item in result["evidence"]:
            source = Path(item["source"]).name
            print(f"- {source} | score={item['score']:.4f}")
            print(f"  {item['text'][:220]}...")

    print(
        "\nDisclaimer: This demo is for educational purposes only and is not medical advice. "
        "Seek professional care for diagnosis or emergencies."
    )


def main() -> None:
    args = parse_args()
    model_bundle = load_model_bundle(args.model_path)
    kb_index_bundle = load_kb_index(args.kb_index_path)

    print("Healthcare Triage Agent Demo")
    while True:
        first_input = input("Start a new case? (y/n): ").strip().lower()
        if first_input in {"n", "no", "exit", "quit"}:
            print("Session ended.")
            return
        if first_input not in {"y", "yes"}:
            print("Please answer y/n.")
            continue

        patient_features = collect_patient_features()
        user_question = input("Follow-up question for KB search (optional): ").strip()
        result = run_triage_agent(
            model_bundle=model_bundle,
            kb_index_bundle=kb_index_bundle,
            patient_features=patient_features,
            user_question=user_question,
            top_k=args.top_k,
        )
        _print_agent_result(result)
        print("\n" + "=" * 72 + "\n")


if __name__ == "__main__":
    main()
