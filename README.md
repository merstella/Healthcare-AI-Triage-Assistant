# Healthcare AI Triage Assistant

A production-oriented machine learning project for emergency triage recommendation, with deterministic safety policy, advanced benchmarking, and an offline RAG + Agent demo.
This is the capstone project I've done to complete the course "Google 5-Day GenAI Intensive" in May 2025. The output of the capstone was a notebook, but I've recently refactored it and optimize the method for better experience.

## Introdction about the project 
[Click here](https://www.linkedin.com/posts/kiran%7Emitra_genai-googlecloud-vertexai-activity-7320829462663557122-3gJb?utm_source=share&utm_medium=member_desktop&rcm=ACoAADknXk0Brrx0NMebd-OsRJU8HGNjRdubwec)

##  Demo CLI
```bash
python3 scripts/demo.py --model-path artifacts/triage_model.joblib
```

## Overview
This project predicts one of three triage outcomes from emergency intake signals:
- `Emergency`
- `Doctor`
- `Self-care`

It combines:
- a supervised classification pipeline,
- a threshold-based safety decision policy (optimized for Emergency recall),
- and an optional retrieval-assisted agent layer for explainability and follow-up support.

## Key Features
- Reproducible CLI workflow: train, evaluate, demo, benchmark.
- Stable global label mapping for robust training/inference consistency.
- Emergency-recall-constrained threshold tuning.
- Advanced model benchmark with:
  - 5-fold `StratifiedKFold` + fixed holdout,
  - text embeddings + tabular features,
  - model family comparison (`LogReg`, `LightGBM`, `CatBoost`),
  - probability calibration (`none`, `sigmoid`, `isotonic`),
  - slice metrics by `sex` and `age_group`.
- Offline RAG + Agent demo with source-grounded evidence retrieval.
- Unit tests and GitHub Actions CI.

## System Design
- Decision layer (deterministic): triage model + tuned emergency threshold policy.
- Knowledge layer (retrieval): TF-IDF KB index over local medical guidance docs.
- Interaction layer: CLI triage demo and CLI agent demo.

## Repository Structure
```text
.
├── docs/
│   └── method_review.md
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── advanced_train.py
│   ├── index_kb.py
│   ├── demo.py
│   └── agent_demo.py
├── kb/
│   └── docs/
├── src/healthcare_ai_agent/
│   ├── constants.py
│   ├── preprocessing.py
│   ├── labeling.py
│   ├── modeling.py
│   ├── inference.py
│   ├── rag.py
│   ├── agent.py
│   ├── recommendation.py
│   └── cli.py
├── tests/
├── requirements.txt
├── requirements-advanced.txt
└── pyproject.toml
```

## Dataset
Public dataset used:
- https://www.kaggle.com/datasets/ilkeryildiz/emergency-service-triage-application

Place the CSV at:
- `data/raw/data.csv`

If you do not have real data yet, generate synthetic demo data:
```bash
python3 scripts/generate_synthetic_data.py --rows 4000 --seed 42 --output-path data/raw/data.csv
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Advanced stack (embeddings + boosting benchmark):
```bash
pip install -r requirements-advanced.txt
```

Optional CPU-only PyTorch:
```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Train Baseline Model
```bash
python3 scripts/train.py \
  --data-path data/raw/data.csv \
  --output-dir artifacts \
  --test-size 0.2 \
  --random-state 42 \
  --emergency-recall-target 0.95 \
  --threshold-grid-size 401 \
  --threshold-tuning-cv-folds 5
```

Outputs:
- `artifacts/triage_model.joblib`
- `artifacts/metrics.json` (primary holdout report)

## Evaluate a Trained Model
```bash
python3 scripts/evaluate.py \
  --model-path artifacts/triage_model.joblib \
  --data-path data/raw/data.csv \
  --output-path artifacts/eval_full_data.json
```

Notes:
- Use `artifacts/metrics.json` from `train.py` as the main holdout performance report.
- `evaluate.py` is useful for sanity-checking the model on any provided dataset.


The demo prints class probabilities and the final recommendation.
The recommendation uses the tuned emergency-threshold policy saved in model metadata.

## Advanced Benchmark
Runs:
- 5-fold `StratifiedKFold` on train split + fixed holdout,
- model family comparison,
- calibration comparison,
- threshold optimization under emergency recall constraint,
- slice metrics by sex/age.

```bash
PYTHONPATH=src python3 scripts/advanced_train.py \
  --data-path data/raw/data.csv \
  --output-path artifacts/advanced_metrics.json \
  --model-output-path artifacts/advanced_model_bundle.joblib \
  --holdout-size 0.2 \
  --random-state 42 \
  --cv-folds 5 \
  --emergency-recall-target 0.95 \
  --threshold-grid-size 401 \
  --fn-cost 10 \
  --fp-cost 1
```

Outputs:
- `artifacts/advanced_metrics.json`
- `artifacts/advanced_model_bundle.joblib`

Method notes: [docs/method_review.md](docs/method_review.md)

## Final Benchmark Snapshot (2026-03-12)
Final full run artifact:
- `artifacts/advanced_metrics_final.json`
- `artifacts/advanced_model_bundle_final.joblib`

Configuration selected by CV:
- Model family: `logreg_embedding`
- Calibration: `none`
- Emergency threshold: `0.1525`

Holdout results (fixed holdout, `n=222`):
- Accuracy: `0.6171`
- Macro-F1: `0.4387`
- Emergency recall: `0.9593`
- Medical cost (`FN=10`, `FP=1`): `118.0`

5-fold CV family comparison (train split):
- `logreg_tfidf`: macro-F1 `0.4627`, emergency recall `0.9511`
- `logreg_embedding`: macro-F1 `0.4692`, emergency recall `0.9511`
- `lightgbm_embedding`: macro-F1 `0.3806`, emergency recall `0.9511`
- `catboost_embedding`: macro-F1 `0.4247`, emergency recall `0.9532`

## Offline RAG + Agent Demo
1. Build KB index:
```bash
python3 scripts/index_kb.py \
  --kb-dir kb/docs \
  --output-path artifacts/kb_index.joblib
```

2. Run agent demo:
```bash
python3 scripts/agent_demo.py \
  --model-path artifacts/triage_model.joblib \
  --kb-index-path artifacts/kb_index.joblib \
  --top-k 3
```

The agent keeps decision authority in the model policy and uses retrieval only for contextual evidence.

## Testing
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## CI
GitHub Actions workflow:
- `.github/workflows/ci.yml`

## GitHub Push
```bash
git init
git add .
git commit -m "Initial production-style healthcare triage project"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Safety and Legal Notice
- This repository is a technical demonstration and not a medical device.
- It must not be used for diagnosis or treatment decisions.
- In real-world scenarios, clinical oversight is mandatory.
