# Project Summary

**Objective:** classify mobile-device records into four encoded `price_range` categories from 20 device specification features.

**Inputs:** 2,000 labeled training records and 1,000 unlabeled inference records.

**Baseline:** `StandardScaler` + `LogisticRegression`, evaluated with a stratified 70/30 holdout and five-fold stratified cross-validation.

**Observed holdout accuracy:** 96.00%.

**Repository assets:** raw CSVs, original Jupyter workfile, reusable Python package, CLI, automated tests, data dictionary, model card, and validation report.

**Important caveat:** upstream data provenance, feature units, and the business meaning of encoded target classes were not supplied. The reported metrics are internal validation results, not independent benchmark results.
