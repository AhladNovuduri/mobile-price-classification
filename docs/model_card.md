# Model Card — Mobile Price Range Classifier

## Model summary

| Item | Description |
|---|---|
| Task | Multiclass classification |
| Target | \`price_range\` (encoded classes 0–3) |
| Estimator | \`LogisticRegression\` |
| Preprocessing | \`StandardScaler\` |
| Implementation | scikit-learn \`Pipeline\` |
| Input | 20 device-specification features |
| Output | Predicted encoded class and class probabilities |

## Intended use

This model is intended as a reproducible educational and portfolio baseline for exploring tabular multiclass classification. It can demonstrate preprocessing, validation, evaluation, and batch inference.

## Out-of-scope use

Do not use this baseline as the sole basis for commercial pricing, financial decisions, consumer eligibility, or production product valuation. The class-to-price mapping and data provenance are not documented.

## Training data

The supplied training dataset contains 2,000 rows and 20 predictors plus the target. Each target class has 500 observations. No missing cells or exact duplicate rows were observed in the supplied training file.

## Evaluation

A stratified 70/30 holdout with \`random_state=42\` was used. Results:

- Accuracy: 0.9600
- Weighted precision: 0.9603
- Weighted recall: 0.9600
- Weighted F1: 0.9600
- Macro one-vs-rest ROC-AUC: 0.9986

Five-fold stratified cross-validation (shuffled, seed 42) yielded accuracy 0.9610 ± 0.0030 and macro-F1 0.9609 ± 0.0029 (mean ± standard deviation across folds).

## Known limitations and risks

- The test file is unlabeled, so its predictive performance cannot be measured here.
- The target class definitions and price boundaries are unknown.
- Feature units, measurement procedures, source population, and collection date are unknown.
- Logistic Regression assumes a linear decision structure in the standardized feature space.
- No fairness, subgroup, temporal, external, or drift evaluation was performed.
- The numeric class order is not explicitly encoded as an ordinal objective.

## Monitoring recommendations

Before any operational use, obtain authoritative data documentation and target definitions; evaluate on a representative external holdout; compare against nonlinear and ordinal alternatives; inspect calibration and subgroup performance; and monitor input drift and class-level errors.


## Additional diagnostics and model comparison

The training CLI now emits a holdout confusion matrix and top-ten permutation importance ranked by decrease in macro-F1. These are diagnostics for the fitted Logistic Regression baseline; permutation importance is model- and sample-specific and must not be interpreted causally.

The `compare` CLI benchmarks Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting with identical stratified five-fold splits, then tunes a small ExtraTrees parameter grid using macro-F1. The tuned candidate is not automatically substituted for the documented baseline. See [Model Comparison and Tuning](model_comparison.md).
