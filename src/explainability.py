"""Feature importance extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def extract_feature_importance(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    top_n: int = 10,
) -> dict[str, float]:
    """Extract feature importance from tree models or permutation importance fallback."""
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = _get_feature_names(preprocessor, X_val)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        result = permutation_importance(
            pipeline,
            X_val,
            y_val,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        importances = result.importances_mean

    pairs = sorted(zip(feature_names, importances), key=lambda item: item[1], reverse=True)
    return {name: float(score) for name, score in pairs[:top_n]}


def _get_feature_names(preprocessor, X: pd.DataFrame) -> list[str]:
    try:
        names = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except Exception:
        return list(X.columns)


def attach_importance_to_best_model(
    trained_models: list[dict[str, Any]],
    best_model_name: str | None,
    splits: dict[str, Any],
    model_metrics: list[Any],
) -> None:
    if not best_model_name:
        return

    trained = next(item for item in trained_models if item["name"] == best_model_name)
    metrics = next(item for item in model_metrics if item.name == best_model_name)
    metrics.feature_importance = extract_feature_importance(
        trained["pipeline"],
        splits["X_val"],
        splits["y_val"],
    )
