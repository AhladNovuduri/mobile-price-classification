"""Command-line interface for model training, comparison, and prediction."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .evaluation import compare_models
from .modeling import (
    TARGET, cross_validation_summary, evaluate_holdout, fit_full_model,
    load_inference_data, load_model, load_training_data, save_json, save_model,
)


def train(args: argparse.Namespace) -> None:
    X, y = load_training_data(args.train)
    _, holdout_metrics = evaluate_holdout(X, y)
    holdout_metrics["cross_validation_5_fold"] = cross_validation_summary(X, y)
    final_model = fit_full_model(X, y)
    save_model(final_model, args.model)
    save_json(holdout_metrics, args.metrics)
    print(f"Model saved to: {args.model}")
    print(f"Metrics saved to: {args.metrics}")
    print(f"Holdout accuracy: {holdout_metrics['accuracy']:.4f}")


def compare(args: argparse.Namespace) -> None:
    X, y = load_training_data(args.train)
    results = compare_models(X, y)
    save_json(results, args.output)
    print(f"Model comparison saved to: {args.output}")
    print(f"Best tuned model: {results['tuning']['best_model']}")


def predict(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    expected_features = list(model.feature_names_in_)
    X, identifiers = load_inference_data(args.input, expected_features)
    predictions = model.predict(X)
    output = pd.DataFrame({TARGET: predictions})
    if identifiers is not None:
        output.insert(0, "id", identifiers.to_numpy())
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(f"Wrote {len(output):,} predictions to: {destination}")

    if args.probabilities_output:
        if not hasattr(model, "predict_proba"):
            raise ValueError("The selected model does not support probability predictions.")
        probabilities = model.predict_proba(X)
        probability_frame = pd.DataFrame(
            probabilities,
            columns=[f"probability_class_{label}" for label in model.classes_],
        )
        probability_frame.insert(0, TARGET, predictions)
        if identifiers is not None:
            probability_frame.insert(0, "id", identifiers.to_numpy())
        probability_path = Path(args.probabilities_output)
        probability_path.parent.mkdir(parents=True, exist_ok=True)
        probability_frame.to_csv(probability_path, index=False)
        print(f"Wrote class probabilities to: {probability_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mobile price-range classification workflow")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Evaluate and fit the baseline model")
    train_parser.add_argument("--train", default="data/raw/train.csv")
    train_parser.add_argument("--model", default="artifacts/price_range_pipeline.joblib")
    train_parser.add_argument("--metrics", default="artifacts/holdout_metrics.json")
    train_parser.set_defaults(func=train)

    compare_parser = commands.add_parser("compare", help="Benchmark and tune candidate models")
    compare_parser.add_argument("--train", default="data/raw/train.csv")
    compare_parser.add_argument("--output", default="artifacts/model_comparison.json")
    compare_parser.set_defaults(func=compare)

    predict_parser = commands.add_parser("predict", help="Predict price-range classes for a CSV")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--output", default="artifacts/predictions.csv")
    predict_parser.add_argument(
        "--probabilities-output",
        help="Optional CSV path for per-class prediction probabilities.",
    )
    predict_parser.set_defaults(func=predict)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
