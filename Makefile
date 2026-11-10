.PHONY: install test train

install:
	python -m pip install -r requirements.txt
	python -m pip install -e .

test:
	pytest

train:
	python -m price_range.cli train --train data/raw/train.csv --model artifacts/price_range_pipeline.joblib --metrics artifacts/holdout_metrics.json
