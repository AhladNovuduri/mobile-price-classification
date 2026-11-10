# Validation Report

## Scope

This report documents integrity checks and baseline evaluation performed on the two supplied CSV files. It does not claim external validation.

## Input integrity

| Check | Training | Test |
|---|---:|---:|
| Rows | 2,000 | 1,000 |
| Columns | 21 | 21 |
| Missing cells | 0 | 0 |
| Exact duplicate rows | 0 | 0 |

The training file contains 20 model features and \`price_range\`. The test file contains \`id\` plus the same 20 features. The feature sets match after excluding \`price_range\` and \`id\`. Test identifiers are unique.

The four target classes are perfectly balanced at 500 records each.

## Holdout protocol

- Split: 70% train / 30% validation
- Stratification: enabled
- Random seed: 42
- Preprocessing: \`StandardScaler\`, fitted inside the pipeline
- Estimator: \`LogisticRegression(max_iter=2000, random_state=42)\`

## Holdout metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9600 |
| Weighted precision | 0.9603 |
| Weighted recall | 0.9600 |
| Weighted F1 | 0.9600 |
| Macro one-vs-rest ROC-AUC | 0.9986 |

### Per-class report

| Class | Precision | Recall | F1 | Support |
|---:|---:|---:|---:|---:|
| 0 | 0.9932 | 0.9667 | 0.9797 | 150 |
| 1 | 0.9412 | 0.9600 | 0.9505 | 150 |
| 2 | 0.9521 | 0.9267 | 0.9392 | 150 |
| 3 | 0.9548 | 0.9867 | 0.9705 | 150 |

## Cross-validation

Five-fold stratified cross-validation used shuffled folds and seed 42.

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Accuracy | 0.9610 | 0.0030 |
| Macro F1 | 0.9609 | 0.0029 |

## Interpretation

The baseline performs strongly on the supplied labeled sample, with consistent fold-level accuracy. The weakest class-level recall in the holdout is class 2 (0.9267). The macro OVR ROC-AUC is high, but it should not be treated as evidence of real-world generalization.

## Reproduction

Install dependencies and run:

\`\`\`bash
pip install -r requirements.txt
pip install -e .
python -m price_range.cli train \\
  --train data/raw/train.csv \\
  --model artifacts/price_range_pipeline.joblib \\
  --metrics artifacts/holdout_metrics.json
pytest
\`\`\`

## Caveats

- The metrics are tied to this exact dataset and split.
- No labeled evaluation set was supplied for the 1,000-row test CSV.
- No tuning, external validation, calibration, fairness analysis, or model comparison is included.
- The original notebook's ROC section uses a single probability column for a multiclass target. The package evaluation instead computes multiclass ROC-AUC with the explicit one-vs-rest strategy.


## Extended diagnostics available in the repository

The reproducible `train` command now also writes a confusion matrix and top-ten permutation-importance summary to the metrics JSON. The matrix is generated from the same stratified holdout split described above; permutation importance is computed on that holdout using macro-F1 as the scoring metric. Exact values are generated at runtime rather than duplicated in this static report.

The `compare` command benchmarks Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting using the same stratified five-fold protocol, and performs a small ExtraTrees grid search. It writes results to `artifacts/model_comparison.json`. See [Model Comparison and Tuning](model_comparison.md).

The repository also includes input-validation tests and a GitHub Actions matrix for Python 3.10, 3.11, and 3.12. The CI workflow should be observed on GitHub before treating the changes as verified in CI.
