"""Project-wide constants."""

from __future__ import annotations

RAW_FEATURE_COLUMNS = [
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
]
RAW_TARGET_COLUMN = "KTAS_expert"

MODEL_FEATURE_COLUMNS = [
    "sex",
    "age_group",
    "arrival_mode",
    "injury",
    "chief_complain",
    "mental_status",
    "pain",
    "hr_group",
    "rr_group",
    "bt_group",
]
MODEL_TARGET_COLUMN = "triage_label"

SEX_MAP = {1: "Female", 2: "Male"}
INJURY_MAP = {1: "No", 2: "Yes"}
PAIN_MAP = {0: "No", 1: "Yes"}
MENTAL_MAP = {
    1: "Alert",
    2: "Verbose Response",
    3: "Pain Response",
    4: "Unresponsive",
}
ARRIVAL_MODE_MAP = {
    1: "Walking",
    2: "Public Ambulance",
    3: "Private Vehicle",
    4: "Private Ambulance",
    5: "Other",
    6: "Other",
    7: "Other",
}
KTAS_TO_TRIAGE_MAP = {
    1: "Emergency",
    2: "Emergency",
    3: "Emergency",
    4: "Doctor",
    5: "Self-care",
}

TRIAGE_LABELS = ["Emergency", "Doctor", "Self-care"]

CATEGORICAL_FEATURE_COLUMNS = [
    "sex",
    "age_group",
    "arrival_mode",
    "injury",
    "mental_status",
    "pain",
    "hr_group",
    "rr_group",
    "bt_group",
]
TEXT_FEATURE_COLUMN = "chief_complain"

