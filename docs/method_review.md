# Methodology Review and Validation Notes

## 1. Objective
Build a safe and reproducible triage recommendation pipeline from emergency intake signals, targeting three operational classes:
- `Emergency`
- `Doctor`
- `Self-care`

Primary optimization priority:
1. maximize `Emergency` recall,
2. then improve macro-F1 under that safety constraint.

## 2. Data and Labeling Design
- Source schema includes demographic, arrival mode, symptom text, mental status, pain, and vital signs.
- `KTAS_expert` is mapped to operational triage classes:
  - 1, 2, 3 -> `Emergency`
  - 4 -> `Doctor`
  - 5 -> `Self-care`
- Stable global label IDs are enforced:
  - `Emergency -> 0`
  - `Doctor -> 1`
  - `Self-care -> 2`

Why this matters:
- prevents class-ID drift between train/eval/inference,
- keeps artifact behavior deterministic.

## 3. Feature Engineering
- Categorical features are normalized and one-hot encoded.
- Vital signs are bucketed into clinically interpretable groups (`Low/Normal/High`).
- `chief_complain` text is used with two alternatives:
  - TF-IDF baseline,
  - multilingual sentence embeddings for advanced benchmark.
- Exact duplicate records are removed after preprocessing to reduce optimistic leakage.

## 4. Baseline Training and Policy
Baseline pipeline:
- `OneHotEncoder` + `TfidfVectorizer` + `LogisticRegression(class_weight='balanced')`.

Decision policy:
- Tune emergency threshold on out-of-fold train predictions.
- Select threshold with emergency recall constraint (`>= target`), then maximize macro-F1.
- Persist selected threshold in model metadata and reuse it at inference time.

## 5. Advanced Evaluation Protocol
Implemented advanced benchmark includes:
- Fixed holdout split for final reporting.
- 5-fold `StratifiedKFold` on train split.
- Model family comparison:
  - `logreg_tfidf`
  - `logreg_embedding`
  - `lightgbm_embedding`
  - `catboost_embedding`
- Calibration comparison (`none`, `sigmoid`, `isotonic`) on selected family.
- Cost-aware threshold search with configurable FN/FP costs.
- Slice metrics by:
  - `sex`
  - `age_group`

Outputs:
- `artifacts/advanced_metrics.json`
- `artifacts/advanced_model_bundle.joblib`

## 6. RAG + Agent Layer (Offline)
The project includes a deterministic agent pattern:
- Decision authority remains in the triage model policy.
- Retrieval layer provides contextual evidence from local knowledge docs.
- This avoids uncontrolled generation changing clinical routing decisions.

Artifacts and scripts:
- KB indexing: `scripts/index_kb.py`
- Agent demo: `scripts/agent_demo.py`

## 7. Known Limitations
- Dataset scope and class balance limit generalization.
- Slice-level metrics on small-support groups can be unstable.
- Current retrieval quality uses TF-IDF; dense retrievers may improve recall.
- This is not validated for real clinical deployment.
