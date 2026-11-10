"""Training and inference utilities for the mobile price-range baseline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "price_range"
IDENTIFIER = "id"
RANDOM_STATE = 42
TEST_SIZE = 0.30


def build_pipeline() -> Pipeline:
    """Return the fixed baseline preprocessing + estimator pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load labeled data and validate its target and feature columns."""
    frame = pd.read_csv(path)
    if TARGET not in frame.columns:
        raise ValueError(f"Training data must contain target column '{TARGET}'.")
    if IDENTIFIER in frame.columns:
        raise ValueError(f"Training data must not contain identifier column '{IDENTIFIER}'.")
    if frame.empty:
        raise ValueError("Training data is empty.")
    if frame[TARGET].isna().any():
        raise ValueError("Training target contains missing values.")
    features = frame.drop(columns=TARGET)
    if features.isna().any().any():
        raise ValueError("Training features contain missing values.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("All model features must be numeric.")
    return features, frame[TARGET]


def load_inference_data(path: str | Path, expected_features: list[str]) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load inference features; preserve an optional id column separately."""
    frame = pd.read_csv(path)
    identifiers = frame[IDENTIFIER].copy() if IDENTIFIER in frame.columns else None
    features = frame.drop(columns=[IDENTIFIER], errors="ignore")
    missing = sorted(set(expected_features) - set(features.columns))
    extra = sorted(set(features.columns) - set(expected_features))
    if missing or extra:
        raise ValueError(f"Inference schema mismatch. Missing={missing}; extra={extra}")
    features = features.loc[:, expected_features]
    if features.isna().any().any():
        raise ValueError("Inference features contain missing values.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("All inference features must be numeric.")
    return features, identifiers


def evaluate_holdout(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict[str, Any]]:
    """Fit on a stratified split and return metrics plus the fitted split model."""
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    predicted = model.predict(X_valid)
    probabilities = model.predict_proba(X_valid)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_valid, predicted, average="weighted", zero_division=0
    )
    metrics: dict[str, Any] = {
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": True},
        "accuracy": float(accuracy_score(y_valid, predicted)),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "macro_ovr_roc_auc": float(
            roc_auc_score(y_valid, probabilities, multi_class="ovr", average="macro")
        ),
        "classification_report": classification_report(
            y_valid, predicted, output_dict=True, zero_division=0
        ),
    }
    return model, metrics


def cross_validation_summary(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, float]]:
    """Return mean and standard deviation for stratified five-fold CV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        build_pipeline(),
        X,
        y,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        n_jobs=-1,
    )
    return {
        metric: {
            "mean": float(results[f"test_{metric}"].mean()),
            "std": float(results[f"test_{metric}"].std()),
        }
        for metric in ("accuracy", "f1_macro")
    }


def fit_full_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Fit the production-for-inference pipeline on all labeled rows."""
    return build_pipeline().fit(X, y)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")


def save_model(model: Pipeline, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def load_model(path: str | Path) -> Pipeline:
    """Load a trusted local joblib artifact. Do not load untrusted pickle files."""
    return joblib.load(path)
