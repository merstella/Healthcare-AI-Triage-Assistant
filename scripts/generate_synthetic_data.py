#!/usr/bin/env python3
"""Generate synthetic triage data compatible with this project schema."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path


SEVERE_COMPLAINTS = [
    "chest pain and shortness of breath",
    "sudden weakness on one side",
    "severe abdominal pain with vomiting",
    "high fever and confusion",
    "uncontrolled bleeding after injury",
    "seizure and loss of consciousness",
    "severe headache with blurred vision",
    "fainting and palpitations",
]

MODERATE_COMPLAINTS = [
    "persistent cough and fever",
    "moderate abdominal discomfort",
    "dizziness and nausea",
    "migraine with light sensitivity",
    "worsening back pain",
    "urinary burning and flank pain",
    "skin infection with swelling",
    "ankle injury with pain",
]

MILD_COMPLAINTS = [
    "mild sore throat",
    "runny nose and cough",
    "minor headache",
    "muscle soreness after exercise",
    "mild stomach upset",
    "small skin rash",
    "low grade fever",
    "mild joint pain",
]


FIELDNAMES = [
    "Sex",
    "Age",
    "Arrival mode",
    "Injury",
    "Chief_complain",
    "Mental",
    "Pain",
    "HR",
    "RR",
    "BT",
    "KTAS_expert",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic triage dataset.")
    parser.add_argument(
        "--rows",
        type=int,
        default=4000,
        help="Number of synthetic rows to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-path",
        default="data/raw/data.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def _bounded_normal(rng: random.Random, mean: float, std: float, low: float, high: float) -> float:
    value = rng.gauss(mean, std)
    return max(low, min(high, value))


def _choice(rng: random.Random, values: list, weights: list[float] | None = None):
    if weights is None:
        return rng.choice(values)
    return rng.choices(values, weights=weights, k=1)[0]


def _sample_emergency(rng: random.Random) -> dict[str, object]:
    return {
        "Sex": int(_choice(rng, [1, 2])),
        "Age": int(_bounded_normal(rng, 58, 18, 18, 95)),
        "Arrival mode": int(_choice(rng, [1, 2, 3, 4, 5, 6, 7], [0.07, 0.45, 0.11, 0.18, 0.06, 0.06, 0.07])),
        "Injury": int(_choice(rng, [1, 2], [0.35, 0.65])),
        "Chief_complain": str(_choice(rng, SEVERE_COMPLAINTS)),
        "Mental": int(_choice(rng, [1, 2, 3, 4], [0.45, 0.25, 0.20, 0.10])),
        "Pain": int(_choice(rng, [0, 1], [0.20, 0.80])),
        "HR": round(_bounded_normal(rng, 112, 24, 35, 195), 1),
        "RR": round(_bounded_normal(rng, 26, 7, 8, 55), 1),
        "BT": round(_bounded_normal(rng, 38.1, 0.9, 34.5, 41.5), 1),
        "KTAS_expert": int(_choice(rng, [1, 2, 3], [0.28, 0.42, 0.30])),
    }


def _sample_doctor(rng: random.Random) -> dict[str, object]:
    return {
        "Sex": int(_choice(rng, [1, 2])),
        "Age": int(_bounded_normal(rng, 45, 17, 16, 90)),
        "Arrival mode": int(_choice(rng, [1, 2, 3, 4, 5, 6, 7], [0.25, 0.16, 0.29, 0.10, 0.07, 0.07, 0.06])),
        "Injury": int(_choice(rng, [1, 2], [0.62, 0.38])),
        "Chief_complain": str(_choice(rng, MODERATE_COMPLAINTS)),
        "Mental": int(_choice(rng, [1, 2, 3, 4], [0.80, 0.13, 0.05, 0.02])),
        "Pain": int(_choice(rng, [0, 1], [0.40, 0.60])),
        "HR": round(_bounded_normal(rng, 92, 17, 40, 170), 1),
        "RR": round(_bounded_normal(rng, 20, 5, 8, 42), 1),
        "BT": round(_bounded_normal(rng, 37.4, 0.7, 34.8, 40.5), 1),
        "KTAS_expert": 4,
    }


def _sample_self_care(rng: random.Random) -> dict[str, object]:
    return {
        "Sex": int(_choice(rng, [1, 2])),
        "Age": int(_bounded_normal(rng, 34, 13, 10, 80)),
        "Arrival mode": int(_choice(rng, [1, 2, 3, 4, 5, 6, 7], [0.52, 0.04, 0.30, 0.03, 0.04, 0.04, 0.03])),
        "Injury": int(_choice(rng, [1, 2], [0.80, 0.20])),
        "Chief_complain": str(_choice(rng, MILD_COMPLAINTS)),
        "Mental": int(_choice(rng, [1, 2, 3, 4], [0.95, 0.04, 0.01, 0.00])),
        "Pain": int(_choice(rng, [0, 1], [0.62, 0.38])),
        "HR": round(_bounded_normal(rng, 80, 11, 45, 130), 1),
        "RR": round(_bounded_normal(rng, 17, 3, 10, 30), 1),
        "BT": round(_bounded_normal(rng, 36.9, 0.4, 35.2, 39.2), 1),
        "KTAS_expert": 5,
    }


def _inject_noise(row: dict[str, object], rng: random.Random) -> None:
    # Add controlled noise so the model cannot overfit trivial rules.
    if rng.random() < 0.08:
        row["HR"] = round(_bounded_normal(rng, 104, 22, 35, 195), 1)
    if rng.random() < 0.08:
        row["RR"] = round(_bounded_normal(rng, 24, 8, 8, 55), 1)
    if rng.random() < 0.08:
        row["BT"] = round(_bounded_normal(rng, 37.8, 0.9, 34.5, 41.5), 1)
    if rng.random() < 0.05:
        row["Mental"] = int(_choice(rng, [1, 2, 3, 4], [0.75, 0.15, 0.07, 0.03]))


def build_dataset(rows: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    records: list[dict[str, object]] = []

    for _ in range(rows):
        group = _choice(rng, ["Emergency", "Doctor", "Self-care"], [0.37, 0.43, 0.20])
        if group == "Emergency":
            row = _sample_emergency(rng)
        elif group == "Doctor":
            row = _sample_doctor(rng)
        else:
            row = _sample_self_care(rng)
        _inject_noise(row, rng)
        records.append(row)

    return records


def write_dataset(records: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="ISO-8859-1", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES, delimiter=";")
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    if args.rows < 100:
        raise ValueError("--rows should be >= 100 for a meaningful demo split.")

    records = build_dataset(args.rows, args.seed)
    output_path = Path(args.output_path)
    write_dataset(records, output_path)

    counts = Counter(int(row["KTAS_expert"]) for row in records)

    print(f"Synthetic dataset saved to: {output_path}")
    print(f"Rows: {len(records)}")
    print("KTAS_expert distribution:")
    for label in sorted(counts):
        ratio = counts[label] / len(records)
        print(f"{label}: {ratio:.3f}")


if __name__ == "__main__":
    main()
