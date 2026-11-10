from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from price_range.modeling import (
    build_pipeline,
    load_inference_data,
    load_training_data,
)

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "raw" / "train.csv"
TEST = ROOT / "data" / "raw" / "test.csv"


def test_supplied_data_integrity():
    X, y = load_training_data(TRAIN)
    X_test, identifiers = load_inference_data(TEST, list(X.columns))
    assert X.shape == (2000, 20)
    assert X_test.shape == (1000, 20)
    assert identifiers is not None and identifiers.is_unique
    assert y.value_counts().sort_index().to_dict() == {0: 500, 1: 500, 2: 500, 3: 500}
    assert not X.isna().any().any()
    assert not X_test.isna().any().any()


def test_pipeline_fits_and_predicts():
    X, y = load_training_data(TRAIN)
    model = build_pipeline().fit(X, y)
    predictions = model.predict(X.iloc[:12])
    probabilities = model.predict_proba(X.iloc[:12])
    assert len(predictions) == 12
    assert probabilities.shape == (12, 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_schema_mismatch_is_rejected(tmp_path):
    frame = pd.read_csv(TEST).drop(columns="wifi")
    path = tmp_path / "bad_test.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_inference_data(path, list(pd.read_csv(TRAIN).drop(columns="price_range").columns))


def test_training_target_required(tmp_path):
    frame = pd.read_csv(TRAIN).drop(columns="price_range")
    path = tmp_path / "bad_train.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="target column"):
        load_training_data(path)
