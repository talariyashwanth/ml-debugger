"""Preprocessing pipeline and data splitting."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import ProblemType
from src.profiling import infer_problem_type


def get_feature_columns(df: pd.DataFrame, target_column: str) -> list[str]:
    return [col for col in df.columns if col != target_column]


def build_preprocessor(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    """Build a leakage-safe preprocessing transformer."""
    numeric_features = [
        col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical_features = [col for col in feature_columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Split data into train, validation, and test sets with stratification when possible."""
    feature_columns = get_feature_columns(df, target_column)
    X = df[feature_columns]
    y = df[target_column]
    problem_type = infer_problem_type(y)

    stratify = y if problem_type != ProblemType.REGRESSION else None
    if stratify is not None and stratify.value_counts().min() < 2:
        stratify = None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    relative_val_size = val_size / (1 - test_size)
    stratify_val = y_temp if problem_type != ProblemType.REGRESSION else None
    if stratify_val is not None and stratify_val.value_counts().min() < 2:
        stratify_val = None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=stratify_val,
    )

    return {
        "feature_columns": feature_columns,
        "problem_type": problem_type,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }
