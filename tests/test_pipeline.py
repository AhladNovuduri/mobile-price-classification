from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from price_range.modeling import (
    build_pipeline, load_inference_data, load_training_data,
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


def test_inference_rejects_infinite_values(tmp_path):
    frame = pd.read_csv(TEST)
    frame.loc[0, "battery_power"] = np.inf
    path = tmp_path / "infinite.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-finite"):
        load_inference_data(path, list(pd.read_csv(TRAIN).drop(columns="price_range").columns))


def test_inference_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("id,battery_power\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_inference_data(path, ["battery_power"])


def test_duplicate_headers_are_rejected(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text("a,a,price_range\n1,2,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column"):
        load_training_data(path)


def test_training_rejects_single_class(tmp_path):
    frame = pd.read_csv(TRAIN)
    frame["price_range"] = 0
    path = tmp_path / "single_class.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="at least two classes"):
        load_training_data(path)


def test_prediction_cli_exports_probabilities(tmp_path):
    from argparse import Namespace
    import joblib
    from price_range.cli import predict

    X, y = load_training_data(TRAIN)
    model = build_pipeline().fit(X, y)
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    predictions_path = tmp_path / "predictions.csv"
    probabilities_path = tmp_path / "probabilities.csv"

    predict(Namespace(
        model=str(model_path),
        input=str(TEST),
        output=str(predictions_path),
        probabilities_output=str(probabilities_path),
    ))

    predictions = pd.read_csv(predictions_path)
    probabilities = pd.read_csv(probabilities_path)
    assert predictions.shape == (1000, 2)
    assert probabilities.shape == (1000, 6)
    assert probabilities["id"].equals(predictions["id"])
    probability_columns = [c for c in probabilities if c.startswith("probability_class_")]
    assert np.allclose(probabilities[probability_columns].sum(axis=1), 1.0)
