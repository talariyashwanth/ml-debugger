"""Dataset profiling and health scoring."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import ID_KEYWORDS, ProblemType
from src.schemas import ColumnProfile, DatasetProfile, TargetProfile


def infer_problem_type(series: pd.Series) -> ProblemType:
    """Infer supervised learning problem type from target column."""
    if pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique(dropna=True)
        if unique_count <= 20 and np.allclose(series.dropna(), series.dropna().astype(int), equal_nan=True):
            if unique_count == 2:
                return ProblemType.BINARY_CLASSIFICATION
            return ProblemType.MULTICLASS_CLASSIFICATION
        return ProblemType.REGRESSION

    unique_count = series.nunique(dropna=True)
    if unique_count == 2:
        return ProblemType.BINARY_CLASSIFICATION
    return ProblemType.MULTICLASS_CLASSIFICATION


def _is_likely_id(name: str, series: pd.Series, row_count: int) -> bool:
    lowered = name.lower()
    if any(keyword in lowered for keyword in ID_KEYWORDS):
        return True
    unique_ratio = series.nunique(dropna=True) / max(row_count, 1)
    return unique_ratio > 0.95 and series.nunique(dropna=True) > 20


def _column_risk(
    missing_pct: float,
    is_constant: bool,
    is_likely_id: bool,
    unique_count: int,
    row_count: int,
) -> str:
    if is_constant or is_likely_id:
        return "high"
    if missing_pct >= 20 or unique_count / max(row_count, 1) > 0.9:
        return "high"
    if missing_pct >= 5 or unique_count > 100:
        return "medium"
    return "low"


def _profile_target(series: pd.Series, problem_type: ProblemType) -> TargetProfile:
    if problem_type == ProblemType.REGRESSION:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((numeric < lower) | (numeric > upper)).sum())
        return TargetProfile(
            problem_type=problem_type,
            mean=float(numeric.mean()),
            median=float(numeric.median()),
            std=float(numeric.std(ddof=0)),
            min_value=float(numeric.min()),
            max_value=float(numeric.max()),
            skew=float(stats.skew(numeric)) if len(numeric) > 2 else 0.0,
            outlier_count=outlier_count,
        )

    value_counts = series.value_counts(dropna=True)
    total = int(value_counts.sum())
    percentages = {str(k): round(v / total * 100, 2) for k, v in value_counts.items()}
    minority_pct = min(percentages.values()) if percentages else 100.0
    return TargetProfile(
        problem_type=problem_type,
        class_counts={str(k): int(v) for k, v in value_counts.items()},
        class_percentages=percentages,
        is_imbalanced=minority_pct < 20.0,
    )


def _compute_health_score(
    missing_pct: float,
    duplicate_pct: float,
    constant_column_count: int,
    likely_id_column_count: int,
    target: TargetProfile | None,
) -> int:
    score = 100.0
    score -= min(missing_pct * 1.5, 30)
    score -= min(duplicate_pct * 2.0, 20)
    score -= min(constant_column_count * 5, 15)
    score -= min(likely_id_column_count * 4, 12)
    if target and target.is_imbalanced:
        score -= 8
    return max(0, min(100, int(round(score))))


def profile_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
    dataset_name: str = "dataset",
) -> DatasetProfile:
    """Profile schema, quality signals, and optional target analysis."""
    if target_column and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    row_count = len(df)
    feature_columns = [col for col in df.columns if col != target_column]
    duplicate_pct = round(df.duplicated().sum() / max(row_count, 1) * 100, 2)
    total_cells = max(row_count * df.shape[1], 1)
    missing_pct = round(df.isna().sum().sum() / total_cells * 100, 2)

    column_profiles: list[ColumnProfile] = []
    constant_column_count = 0
    likely_id_column_count = 0
    numeric_feature_count = 0
    categorical_feature_count = 0

    for column in df.columns:
        if column == target_column:
            continue

        series = df[column]
        missing_pct_col = round(series.isna().mean() * 100, 2)
        unique_count = int(series.nunique(dropna=True))
        is_constant = unique_count <= 1
        is_near_constant = unique_count <= max(2, int(row_count * 0.01))
        is_likely_id = _is_likely_id(column, series, row_count)
        risk = _column_risk(missing_pct_col, is_constant, is_likely_id, unique_count, row_count)

        if is_constant:
            constant_column_count += 1
        if is_likely_id:
            likely_id_column_count += 1
        if pd.api.types.is_numeric_dtype(series):
            numeric_feature_count += 1
        else:
            categorical_feature_count += 1

        sample_values = series.dropna().head(3).tolist()
        column_profiles.append(
            ColumnProfile(
                name=column,
                dtype=str(series.dtype),
                missing_pct=missing_pct_col,
                unique_count=unique_count,
                is_constant=is_constant,
                is_near_constant=is_near_constant,
                is_likely_id=is_likely_id,
                risk=risk,
                sample_values=sample_values,
            )
        )

    target_profile = None
    if target_column:
        problem_type = infer_problem_type(df[target_column])
        target_profile = _profile_target(df[target_column], problem_type)

    health_score = _compute_health_score(
        missing_pct,
        duplicate_pct,
        constant_column_count,
        likely_id_column_count,
        target_profile,
    )

    return DatasetProfile(
        dataset_name=dataset_name,
        row_count=row_count,
        column_count=df.shape[1],
        feature_count=len(feature_columns),
        missing_pct=missing_pct,
        duplicate_pct=duplicate_pct,
        numeric_feature_count=numeric_feature_count,
        categorical_feature_count=categorical_feature_count,
        constant_column_count=constant_column_count,
        likely_id_column_count=likely_id_column_count,
        health_score=health_score,
        columns=column_profiles,
        target=target_profile,
    )
