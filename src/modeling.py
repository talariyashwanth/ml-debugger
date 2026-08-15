"""Baseline model training."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from src.constants import ProblemType
from src.preprocessing import build_preprocessor


def get_baseline_models(problem_type: ProblemType) -> dict[str, Any]:
    if problem_type == ProblemType.REGRESSION:
        return {
            "Dummy": DummyRegressor(strategy="median"),
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
        }

    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
    }


def train_baseline_models(
    splits: dict[str, Any],
    df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Train baseline models using sklearn Pipelines."""
    feature_columns = splits["feature_columns"]
    problem_type = splits["problem_type"]
    preprocessor = build_preprocessor(df, feature_columns)
    models = get_baseline_models(problem_type)

    trained: list[dict[str, Any]] = []
    for name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(splits["X_train"], splits["y_train"])
        trained.append(
            {
                "name": name,
                "pipeline": pipeline,
                "problem_type": problem_type,
            }
        )
    return trained
