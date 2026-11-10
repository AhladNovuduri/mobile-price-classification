"""Cross-validated model benchmarking and lightweight hyperparameter tuning."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from .modeling import RANDOM_STATE, build_pipeline


def compare_models(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Compare baseline and tree models, then tune ExtraTrees using macro-F1."""
    if y.value_counts().min() < 5:
        raise ValueError("Each target class needs at least five rows for 5-fold CV.")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "logistic_regression": build_pipeline(LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        )),
        "random_forest": build_pipeline(RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )),
        "extra_trees": build_pipeline(ExtraTreesClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        )),
        "hist_gradient_boosting": build_pipeline(HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        )),
    }
    scores: dict[str, Any] = {}
    for name, estimator in candidates.items():
        result = cross_validate(
            estimator, X, y, cv=cv,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            n_jobs=-1, return_train_score=False,
        )
        scores[name] = {
            metric: {
                "mean": float(result[f"test_{metric}"].mean()),
                "std": float(result[f"test_{metric}"].std()),
            }
            for metric in ("accuracy", "f1_macro")
        }

    search = GridSearchCV(
        build_pipeline(ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        param_grid={
            "classifier__n_estimators": [200, 400],
            "classifier__max_features": ["sqrt", 1.0],
            "classifier__min_samples_leaf": [1, 2],
        },
        scoring="f1_macro", cv=cv, n_jobs=-1, refit=True,
    )
    search.fit(X, y)
    return {
        "cross_validation": {
            "folds": 5, "shuffle": True, "random_state": RANDOM_STATE,
            "scoring": ["accuracy", "f1_macro"],
        },
        "candidate_models": scores,
        "tuning": {
            "best_model": "ExtraTreesClassifier",
            "objective": "f1_macro",
            "best_cv_f1_macro": float(search.best_score_),
            "best_parameters": search.best_params_,
            "grid_candidates": len(search.cv_results_["params"]),
        },
        "notes": [
            "All scores are cross-validation estimates on the supplied labeled data.",
            "The tuned estimator is not automatically promoted; compare results before changing the documented baseline.",
        ],
    }
