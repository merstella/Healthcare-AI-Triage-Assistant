.PHONY: synthetic-data train evaluate demo test advanced-train index-kb agent-demo

synthetic-data:
	python3 scripts/generate_synthetic_data.py --rows 4000 --seed 42 --output-path data/raw/data.csv

train:
	python3 scripts/train.py --data-path data/raw/data.csv --output-dir artifacts --test-size 0.2 --random-state 42

advanced-train:
	PYTHONPATH=src python3 scripts/advanced_train.py --data-path data/raw/data.csv --output-path artifacts/advanced_metrics.json --model-output-path artifacts/advanced_model_bundle.joblib --holdout-size 0.2 --random-state 42 --cv-folds 5 --emergency-recall-target 0.95 --threshold-grid-size 401 --fn-cost 10 --fp-cost 1

evaluate:
	python3 scripts/evaluate.py --model-path artifacts/triage_model.joblib --data-path data/raw/data.csv --output-path artifacts/eval_full_data.json

demo:
	python3 scripts/demo.py --model-path artifacts/triage_model.joblib

index-kb:
	python3 scripts/index_kb.py --kb-dir kb/docs --output-path artifacts/kb_index.joblib

agent-demo:
	python3 scripts/agent_demo.py --model-path artifacts/triage_model.joblib --kb-index-path artifacts/kb_index.joblib

test:
	python3 -m unittest discover -s tests -p "test_*.py"
