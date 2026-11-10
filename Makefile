.PHONY: install test train compare

install:
	python -m pip install -r requirements.txt
	python -m pip install -e ".[dev]"

test:
	pytest

train:
	python -m price_range.cli train --train data/raw/train.csv --model artifacts/price_range_pipeline.joblib --metrics artifacts/holdout_metrics.json

compare:
	python -m price_range.cli compare --train data/raw/train.csv --output artifacts/model_comparison.json
