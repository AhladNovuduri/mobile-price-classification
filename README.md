# Mobile Price Range Classification

A reproducible multiclass machine-learning workflow for classifying mobile devices into four encoded price-range categories.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E?logo=scikitlearn&logoColor=white)
![CI](https://github.com/AhladNovuduri/mobile-price-classification/actions/workflows/tests.yml/badge.svg)

## Project overview

The repository contains the supplied CSV data, an original notebook, a reusable scikit-learn pipeline, automated tests, evaluation and model-comparison commands, and supporting documentation.

The documented baseline is **StandardScaler + LogisticRegression**. Preprocessing is fitted inside the pipeline to avoid leakage between training and validation folds.

## Results

The baseline's stratified 70/30 holdout evaluation (random state 42) on the supplied training data reported:

| Metric | Result |
|---|---:|
| Accuracy | **0.9600** |
| Weighted precision | **0.9603** |
| Weighted recall | **0.9600** |
| Weighted F1 | **0.9600** |
| Macro one-vs-rest ROC-AUC | **0.9986** |

Five-fold stratified cross-validation reported accuracy **0.9610 ± 0.0030** and macro-F1 **0.9609 ± 0.0029** (mean ± standard deviation).

These are internal estimates on the supplied labeled data, not performance on the unlabeled test file or an external dataset. See [the validation report](docs/validation_report.md).

## Repository structure

```text
.github/workflows/       # Continuous integration
data/raw/                # Supplied train.csv and test.csv
docs/                    # Data dictionary, model card, validation, comparison
notebooks/               # Original supplied notebook
src/price_range/         # Reusable modeling, evaluation, and CLI code
tests/                   # Unit and data-integrity tests
```

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Train and evaluate the baseline

```bash
price-range train \
  --train data/raw/train.csv \
  --model artifacts/price_range_pipeline.joblib \
  --metrics artifacts/holdout_metrics.json
```

The training report includes holdout metrics, a confusion matrix, a per-class classification report, top-ten permutation importance, and five-fold cross-validation. Generated files are written to `artifacts/`, which is ignored by Git.

### Compare and tune candidate models

```bash
price-range compare \
  --train data/raw/train.csv \
  --output artifacts/model_comparison.json
```

This benchmarks Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting with the same stratified folds, then performs a small ExtraTrees grid search. See [Model Comparison](docs/model_comparison.md).

### Predict labels and optional probabilities

```bash
price-range predict \
  --model artifacts/price_range_pipeline.joblib \
  --input data/raw/test.csv \
  --output artifacts/test_predictions.csv \
  --probabilities-output artifacts/test_probabilities.csv
```

The test file's `id` column is retained in outputs and excluded from model features. The supplied input files are never overwritten.

### Run tests

```bash
pytest
```

GitHub Actions runs the same suite on Python 3.10, 3.11, and 3.12 for pushes and pull requests.

## Data

| File | Rows | Columns | Role |
|---|---:|---:|---|
| `data/raw/train.csv` | 2,000 | 21 | Features plus target |
| `data/raw/test.csv` | 1,000 | 21 | Identifier plus features |

The training target has four encoded classes (0–3), with 500 rows per class. The test set has no target labels. The supplied data has no missing values or exact duplicate rows. Feature descriptions and limitations are in the [data dictionary](docs/data_dictionary.md).

**Provenance:** the files were supplied with this project; no upstream source, collection protocol, feature-unit definitions, or data license accompanied them.

## Methodology and safeguards

1. Validate CSV headers, target, feature schema, numeric types, missing values, and finite values.
2. Split labeled data using stratified sampling and a fixed random seed.
3. Fit preprocessing and the estimator together in a scikit-learn pipeline.
4. Report class-level metrics and confusion matrix alongside aggregate scores.
5. Use stratified cross-validation for stability checks and identical folds for candidate comparisons.
6. Refit the documented baseline on all labeled rows before inference.

## Limitations

- No labels are available for the 1,000-row test file, so its predictive performance cannot be measured.
- The encoded class boundaries and currency thresholds are unknown.
- Feature units, source population, and collection date are undocumented.
- Cross-validation and holdout results are internal estimates, not external validation.
- Permutation importance is a model-specific diagnostic, not causal evidence.
- No fairness, temporal, external, or drift evaluation has been performed.
- This is an educational/portfolio baseline, not a production pricing or valuation system.

## Project documentation

- [Data dictionary](docs/data_dictionary.md)
- [Model card](docs/model_card.md)
- [Validation report](docs/validation_report.md)
- [Model comparison and tuning](docs/model_comparison.md)
- [CI workflow notes](.github/workflows/README.md)

## License

No license was supplied with the project. Reuse permissions are therefore not specified; add a license only after choosing the terms you want to grant.
