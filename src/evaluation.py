"""Model evaluation, overfitting, and distribution shift detection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.constants import ProblemType, Severity
from src.schemas import DiagnosticIssue, ModelMetrics


def _safe_roc_auc(y_true, y_pred_proba, problem_type: ProblemType) -> float | None:
    if problem_type == ProblemType.REGRESSION:
        return None
    try:
        if problem_type == ProblemType.BINARY_CLASSIFICATION:
            if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                return float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            return float(roc_auc_score(y_true, y_pred_proba))
        classes = np.unique(y_true)
        y_bin = label_binarize(y_true, classes=classes)
        return float(roc_auc_score(y_bin, y_pred_proba, multi_class="ovr", average="weighted"))
    except Exception:
        return None


def _classification_metrics(y_true, y_pred, y_pred_proba, problem_type: ProblemType) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    roc_auc = _safe_roc_auc(y_true, y_pred_proba, problem_type)
    if roc_auc is not None:
        metrics["roc_auc"] = roc_auc
    return metrics


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(
    pipeline,
    X_train,
    y_train,
    X_val,
    y_val,
    problem_type: ProblemType,
    name: str,
) -> ModelMetrics:
    train_pred = pipeline.predict(X_train)
    val_pred = pipeline.predict(X_val)

    if problem_type == ProblemType.REGRESSION:
        train_metrics = _regression_metrics(y_train, train_pred)
        val_metrics = _regression_metrics(y_val, val_pred)
        primary_metric = "r2"
    else:
        train_proba = _predict_proba_safe(pipeline, X_train)
        val_proba = _predict_proba_safe(pipeline, X_val)
        train_metrics = _classification_metrics(y_train, train_pred, train_proba, problem_type)
        val_metrics = _classification_metrics(y_val, val_pred, val_proba, problem_type)
        primary_metric = "f1"

    cm = None
    if problem_type != ProblemType.REGRESSION:
        cm = confusion_matrix(y_val, val_pred).tolist()

    return ModelMetrics(
        name=name,
        metrics={**val_metrics, "primary_metric": primary_metric},
        train_metrics=train_metrics,
        confusion_matrix=cm,
    )


def _predict_proba_safe(pipeline, X):
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)
    preds = pipeline.predict(X)
    return np.column_stack([1 - preds, preds])


def evaluate_all_models(
    trained_models: list[dict[str, Any]],
    splits: dict[str, Any],
) -> list[ModelMetrics]:
    results = []
    for item in trained_models:
        results.append(
            evaluate_model(
                item["pipeline"],
                splits["X_train"],
                splits["y_train"],
                splits["X_val"],
                splits["y_val"],
                item["problem_type"],
                item["name"],
            )
        )
    return results


def select_best_model(models: list[ModelMetrics], problem_type: ProblemType) -> str | None:
    if not models:
        return None
    metric = "r2" if problem_type == ProblemType.REGRESSION else "f1"
    reverse = problem_type == ProblemType.REGRESSION
    return sorted(models, key=lambda m: m.metrics.get(metric, float("-inf")), reverse=reverse)[0].name


def detect_overfitting(model: ModelMetrics, problem_type: ProblemType) -> DiagnosticIssue | None:
    if not model.train_metrics:
        return None
    metric = "r2" if problem_type == ProblemType.REGRESSION else "f1"
    train_score = model.train_metrics.get(metric)
    val_score = model.metrics.get(metric)
    if train_score is None or val_score is None:
        return None

    if problem_type == ProblemType.REGRESSION:
        gap = train_score - val_score
        threshold = 0.15
        gap_pct = gap * 100
    else:
        gap = train_score - val_score
        threshold = 0.15
        gap_pct = gap * 100

    if gap <= threshold:
        return None

    return DiagnosticIssue(
        id=f"overfitting_{model.name.lower().replace(' ', '_')}",
        title=f"Possible overfitting in {model.name}",
        category="generalization",
        severity=Severity.HIGH if gap >= 0.25 else Severity.MEDIUM,
        evidence=f"Train {metric.upper()} = {train_score:.2f}, validation {metric.upper()} = {val_score:.2f}",
        explanation=f"Validation performance is materially lower than training performance (gap: {gap_pct:.1f} points).",
        recommendation="Investigate model complexity, regularization, data quantity, and feature engineering.",
        confidence=0.82,
    )


def detect_underfitting(model: ModelMetrics, problem_type: ProblemType) -> DiagnosticIssue | None:
    if not model.train_metrics:
        return None
    metric = "r2" if problem_type == ProblemType.REGRESSION else "f1"
    train_score = model.train_metrics.get(metric)
    val_score = model.metrics.get(metric)
    if train_score is None or val_score is None:
        return None

    if problem_type == ProblemType.REGRESSION:
        low_performance = val_score < 0.3
        small_gap = abs(train_score - val_score) < 0.05
    else:
        low_performance = val_score < 0.6
        small_gap = abs(train_score - val_score) < 0.05

    if not (low_performance and small_gap):
        return None

    return DiagnosticIssue(
        id=f"underfitting_{model.name.lower().replace(' ', '_')}",
        title=f"Possible underfitting in {model.name}",
        category="generalization",
        severity=Severity.MEDIUM,
        evidence=f"Train {metric.upper()} = {train_score:.2f}, validation {metric.upper()} = {val_score:.2f}",
        explanation="Both training and validation performance are weak with a small generalization gap.",
        recommendation="Investigate weak features, insufficient model complexity, or poor representation.",
        confidence=0.75,
    )


def detect_distribution_shift(
    splits: dict[str, Any],
    feature_columns: list[str],
) -> list[DiagnosticIssue]:
    issues: list[DiagnosticIssue] = []
    X_train = splits["X_train"]
    X_test = splits["X_test"]

    for column in feature_columns:
        train_series = X_train[column]
        test_series = X_test[column]
        if pd.api.types.is_numeric_dtype(train_series):
            train_mean = float(pd.to_numeric(train_series, errors="coerce").mean())
            test_mean = float(pd.to_numeric(test_series, errors="coerce").mean())
            denom = abs(train_mean) if train_mean != 0 else 1.0
            relative_shift = abs(test_mean - train_mean) / denom
            if relative_shift >= 0.25:
                issues.append(
                    DiagnosticIssue(
                        id=f"shift_{column}",
                        title=f"Distribution shift in {column}",
                        category="distribution_shift",
                        severity=Severity.HIGH if relative_shift >= 0.5 else Severity.MEDIUM,
                        evidence=f"Train mean: {train_mean:.2f}, test mean: {test_mean:.2f}",
                        explanation="Train and test distributions differ materially for this feature.",
                        recommendation="Inspect sampling strategy and whether test data comes from a different population.",
                        confidence=0.7,
                        feature=column,
                    )
                )
        else:
            train_dist = train_series.astype(str).value_counts(normalize=True)
            test_dist = test_series.astype(str).value_counts(normalize=True)
            categories = set(train_dist.index) | set(test_dist.index)
            l1_shift = sum(abs(train_dist.get(cat, 0.0) - test_dist.get(cat, 0.0)) for cat in categories)
            if l1_shift >= 0.35:
                issues.append(
                    DiagnosticIssue(
                        id=f"shift_{column}",
                        title=f"Categorical distribution shift in {column}",
                        category="distribution_shift",
                        severity=Severity.HIGH if l1_shift >= 0.6 else Severity.MEDIUM,
                        evidence=f"L1 category shift = {l1_shift:.2f}",
                        explanation="Category proportions differ between train and test splits.",
                        recommendation="Review whether the split reflects the deployment population.",
                        confidence=0.68,
                        feature=column,
                    )
                )
    return issues
