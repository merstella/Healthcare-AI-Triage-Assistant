"""Interactive CLI for recruiter demo."""

from __future__ import annotations

from pathlib import Path

from .inference import (
    get_emergency_threshold_from_bundle,
    load_model_bundle,
    predict_from_features,
    predict_label_from_features,
)
from .preprocessing import bucket_age, bucket_bt, bucket_hr, bucket_rr
from .recommendation import make_recommendation


ARRIVAL_MODE_OPTIONS = [
    "Walking",
    "Public Ambulance",
    "Private Vehicle",
    "Private Ambulance",
    "Other",
]
MENTAL_STATUS_OPTIONS = [
    "Alert",
    "Verbose Response",
    "Pain Response",
    "Unresponsive",
]


def _ask_choice(prompt: str, options: list[str]) -> str:
    print(prompt)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    while True:
        raw = input("> ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        for option in options:
            if raw.lower() == option.lower():
                return option
        print("Invalid value, try again.")


def _ask_float(prompt: str, *, minimum: float, maximum: float) -> float:
    while True:
        raw = input(f"{prompt}: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"Value must be between {minimum} and {maximum}.")


def _ask_yes_no(prompt: str) -> str:
    while True:
        raw = input(f"{prompt} (y/n): ").strip().lower()
        if raw in {"y", "yes"}:
            return "Yes"
        if raw in {"n", "no"}:
            return "No"
        print("Please enter y/n.")


def _ask_sex() -> str:
    while True:
        raw = input("Sex (female/male): ").strip().lower()
        if raw in {"female", "f"}:
            return "Female"
        if raw in {"male", "m"}:
            return "Male"
        print("Please type female or male.")


def collect_patient_features() -> dict[str, str]:
    sex = _ask_sex()
    age = _ask_float("Age", minimum=0.0, maximum=120.0)
    arrival_mode = _ask_choice("Arrival mode:", ARRIVAL_MODE_OPTIONS)
    injury = _ask_yes_no("Any injury")
    chief_complain = input("Chief complaint (free text): ").strip()
    mental_status = _ask_choice("Mental status:", MENTAL_STATUS_OPTIONS)
    pain = _ask_yes_no("Any pain")
    hr = _ask_float("Heart rate (HR)", minimum=1.0, maximum=260.0)
    rr = _ask_float("Respiratory rate (RR)", minimum=1.0, maximum=100.0)
    bt = _ask_float("Body temperature (BT, Celsius)", minimum=30.0, maximum=45.0)

    return {
        "sex": sex,
        "age_group": bucket_age(age),
        "arrival_mode": arrival_mode,
        "injury": injury,
        "chief_complain": chief_complain if chief_complain else "unspecified complaint",
        "mental_status": mental_status,
        "pain": pain,
        "hr_group": bucket_hr(hr),
        "rr_group": bucket_rr(rr),
        "bt_group": bucket_bt(bt),
    }


def _print_result(probabilities: dict[str, float], recommendation: str, rationale: str) -> None:
    print("\nAutomated probabilities:")
    for label in ("Emergency", "Doctor", "Self-care"):
        print(f"- {label}: {probabilities[label] * 100:.2f}%")

    print(f"\nRecommended next step: {recommendation}")
    print(f"Reason: {rationale}")
    print(
        "\nDisclaimer: This tool is for educational/demo use only and does not replace "
        "professional diagnosis. If symptoms are severe or worsening, seek medical help immediately."
    )


def run_interactive_demo(model_path: str | Path) -> None:
    bundle = load_model_bundle(model_path)
    tuned_threshold = get_emergency_threshold_from_bundle(bundle)
    print("Healthcare Triage Demo")
    print("Type 'exit' at the first question to quit.\n")

    while True:
        first_input = input("Start a new case? (y/n): ").strip().lower()
        if first_input in {"n", "no", "exit", "quit"}:
            print("Session ended.")
            return
        if first_input not in {"y", "yes"}:
            print("Please answer y/n.")
            continue

        patient_features = collect_patient_features()
        probabilities = predict_from_features(bundle, patient_features)
        if tuned_threshold is not None:
            label, applied_threshold = predict_label_from_features(
                bundle,
                patient_features,
                emergency_threshold=tuned_threshold,
            )
            rationale = (
                "Recommendation uses tuned emergency-threshold policy "
                f"(threshold={applied_threshold:.3f}) from model metadata."
            )
            _print_result(probabilities, label, rationale)
        else:
            recommendation = make_recommendation(probabilities)
            _print_result(recommendation.probabilities, recommendation.level, recommendation.rationale)
        print("\n" + "=" * 72 + "\n")
