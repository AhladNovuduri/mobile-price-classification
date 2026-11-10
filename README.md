<div align="center">

# Mobile Price Range Classification

**A reproducible multiclass machine-learning workflow for classifying mobile devices into price-range categories.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipeline-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-Not%20Specified-lightgrey)
![Status](https://img.shields.io/badge/Status-Research%20Baseline-blue)

</div>

---

## Project overview

This repository contains a complete baseline workflow for predicting a device's encoded `price_range` class from its hardware and connectivity specifications. It includes the supplied train/test CSV files, the original Jupyter workfile, a reusable scikit-learn pipeline, automated tests, and project documentation.

The baseline estimator is **StandardScaler + LogisticRegression**. The pipeline keeps preprocessing and estimation together to reduce train/validation leakage and make inference consistent.

## Results at a glance

A reproducible stratified 70/30 holdout evaluation (random state `42`) on the supplied training file produced:

| Metric | Result |
|---|---:|
| Holdout accuracy | **0.9600** |
| Weighted precision | **0.9603** |
| Weighted recall | **0.9600** |
| Weighted F1 | **0.9600** |
| Macro one-vs-rest ROC-AUC | **0.9986** |

Five-fold stratified cross-validation yielded mean accuracy **0.9610** (standard deviation **0.0030**) and mean macro-F1 **0.9609** (standard deviation **0.0029**).

These are baseline results on the provided training data, not a score on a hidden benchmark or the unlabeled test set. See [the validation report](docs/validation_report.md) for methodology and caveats.

## Repository structure

```text
mobile-price-classification/
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
├── docs/
│   ├── data_dictionary.md
│   ├── model_card.md
│   └── validation_report.md
├── notebooks/
│   └── predictor.ipynb       # Original supplied workfile; preserved unchanged
├── src/
│   └── price_range/
│       ├── __init__.py
│       ├── modeling.py
│       └── cli.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── CONTRIBUTING.md
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Quick start

### 1. Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
.venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Train and evaluate

Run from the repository root:

```bash
python -m price_range.cli train \\
  --train data/raw/train.csv \\
  --model artifacts/price_range_pipeline.joblib \\
  --metrics artifacts/holdout_metrics.json
```

### 3. Generate predictions

The test file contains an `id` column that is excluded from model features. Predictions retain the identifier and are written to a new output file:

```bash
python -m price_range.cli predict \\
  --model artifacts/price_range_pipeline.joblib \\
  --input data/raw/test.csv \\
  --output artifacts/test_predictions.csv
```

The CLI does not overwrite the supplied CSV files. Use a new output path.

### 4. Run quality checks

```bash
pytest
```

Or use `make test` where Make is available.

## Data

| File | Rows | Columns | Role |
|---|---:|---:|---|
| `data/raw/train.csv` | 2,000 | 21 | Features plus target |
| `data/raw/test.csv` | 1,000 | 21 | Identifier plus features |

The training target is `price_range` with four equally represented encoded classes (`0`–`3`, 500 rows each). The test set does not include the target. Both supplied files have no missing values or exact duplicate rows. See [Data Dictionary](docs/data_dictionary.md).

**Data provenance:** the dataset files are included in this repository. An upstream source, collection protocol, licensing terms, and feature units were not provided; this repository does not claim independent verification of those details.

## Methodology

1. Validate the training/test schema and basic data integrity.
2. Separate the target (`price_range`) from the 20 predictors.
3. Split the labeled data with a stratified 70/30 holdout and fixed seed.
4. Fit `StandardScaler` and `LogisticRegression` in one scikit-learn `Pipeline`.
5. Report accuracy, weighted precision/recall/F1, macro one-vs-rest ROC-AUC, and a per-class classification report.
6. Use stratified five-fold cross-validation as a stability check.
7. Refit on the full labeled dataset before generating test-set predictions.

The model and validation details are documented in the [Model Card](docs/model_card.md) and [Validation Report](docs/validation_report.md).

## Notebook

`notebooks/predictor.ipynb` is the original notebook included in the repository. It is retained as a workfile and has not been rewritten. For a repeatable command-line workflow, use the package under `src/`.

## Limitations

- The reported evaluation is an internal holdout and cross-validation estimate; no external test labels are available.
- The target classes are treated as nominal multiclass labels by Logistic Regression. Their numeric ordering is not explicitly modeled.
- No hyperparameter search or model comparison is claimed.
- Feature units and dataset provenance are not documented in the supplied files.
- The baseline is not a production deployment or a pricing recommendation system.

## Reproducibility

The random seed, split strategy, model configuration, dependencies, and CLI commands are explicit. The original input assets are preserved under `data/raw/`; generated artifacts are written to `artifacts/`, which is excluded from version control by default.

## Maintainer

Maintained by **Ahlad Novuduri**.

---
