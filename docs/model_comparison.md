# Model Comparison and Tuning

## Purpose

Compare the Logistic Regression baseline with tree-based alternatives using identical stratified five-fold splits. The comparison is intended to test whether nonlinear decision boundaries improve macro-averaged class performance.

## Reproduce

```bash
python -m price_range.cli compare \
  --train data/raw/train.csv \
  --output artifacts/model_comparison.json
```

The command evaluates Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting using accuracy and macro-F1. It then performs a small grid search over ExtraTrees hyperparameters using macro-F1 as the selection objective.

## Interpretation rules

- Compare mean scores and fold-to-fold standard deviations, not only the single best fold.
- Prefer macro-F1 when class-level performance is important; inspect the confusion matrix and per-class report before selecting a model.
- The supplied dataset is balanced, but macro-F1 remains useful for detecting uneven class performance.
- The tuned model is not automatically promoted to the default. A candidate should replace the baseline only after a documented, repeatable improvement and a fresh holdout evaluation.
- Cross-validation on this dataset is internal validation, not external evidence of real-world performance.

## Feature interpretation

The training command reports the top ten permutation-importance features on the holdout set. Importance is measured as the decrease in macro-F1 after shuffling one feature. It is a model- and sample-specific diagnostic, not a causal effect or a statement about market price.
