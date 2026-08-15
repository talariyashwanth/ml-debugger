"""Data quality issue detection."""

from __future__ import annotations

import pandas as pd

from src.constants import Severity
from src.schemas import ColumnProfile, DatasetProfile, DiagnosticIssue


def detect_quality_issues(profile: DatasetProfile) -> list[DiagnosticIssue]:
    """Detect missing values, duplicates, constants, IDs, and cardinality risks."""
    issues: list[DiagnosticIssue] = []

    if profile.missing_pct > 0:
        severity = Severity.HIGH if profile.missing_pct >= 20 else Severity.MEDIUM
        issues.append(
            DiagnosticIssue(
                id="missing_values_overall",
                title="Missing values detected",
                category="data_quality",
                severity=severity,
                evidence=f"Overall missing values: {profile.missing_pct:.1f}%",
                explanation="Missing values can reduce effective sample size and bias model training.",
                recommendation="Review imputation strategy and investigate why values are missing.",
                confidence=0.95,
            )
        )

    if profile.duplicate_pct > 0:
        severity = Severity.HIGH if profile.duplicate_pct >= 5 else Severity.MEDIUM
        issues.append(
            DiagnosticIssue(
                id="duplicate_rows",
                title="Duplicate rows detected",
                category="data_quality",
                severity=severity,
                evidence=f"Duplicate rows: {profile.duplicate_pct:.1f}%",
                explanation="Duplicates can inflate performance and distort validation splits.",
                recommendation="Remove or deduplicate rows before training and evaluation.",
                confidence=0.9,
            )
        )

    for column in profile.columns:
        issues.extend(_column_quality_issues(column))

    return issues


def _column_quality_issues(column: ColumnProfile) -> list[DiagnosticIssue]:
    issues: list[DiagnosticIssue] = []

    if column.is_constant:
        issues.append(
            DiagnosticIssue(
                id=f"constant_{column.name}",
                title=f"Constant column: {column.name}",
                category="data_quality",
                severity=Severity.MEDIUM,
                evidence=f"{column.name} has a single unique value.",
                explanation="Constant features provide no predictive signal.",
                recommendation=f"Drop '{column.name}' from the feature set.",
                confidence=0.98,
                feature=column.name,
            )
        )

    if column.is_likely_id:
        issues.append(
            DiagnosticIssue(
                id=f"likely_id_{column.name}",
                title=f"Likely identifier column: {column.name}",
                category="data_quality",
                severity=Severity.HIGH,
                evidence=f"{column.name} has {column.unique_count} unique values.",
                explanation="Identifier columns often cause overfitting and do not generalize.",
                recommendation=f"Exclude '{column.name}' from model features unless it is intentionally predictive.",
                confidence=0.85,
                feature=column.name,
            )
        )

    if column.missing_pct >= 5:
        severity = Severity.HIGH if column.missing_pct >= 20 else Severity.MEDIUM
        issues.append(
            DiagnosticIssue(
                id=f"missing_{column.name}",
                title=f"Missing values in {column.name}",
                category="data_quality",
                severity=severity,
                evidence=f"Missing: {column.missing_pct:.1f}%",
                explanation="Feature-level missingness may require targeted imputation or removal.",
                recommendation=f"Inspect missingness pattern for '{column.name}' and choose an imputation strategy.",
                confidence=0.9,
                feature=column.name,
            )
        )

    if column.unique_count > 100 and not column.is_likely_id:
        issues.append(
            DiagnosticIssue(
                id=f"high_cardinality_{column.name}",
                title=f"High cardinality feature: {column.name}",
                category="data_quality",
                severity=Severity.MEDIUM,
                evidence=f"{column.unique_count} unique values in {column.name}.",
                explanation="High-cardinality categoricals can increase sparsity and overfitting risk.",
                recommendation=f"Consider grouping, target encoding, or feature hashing for '{column.name}'.",
                confidence=0.75,
                feature=column.name,
            )
        )

    return issues


def detect_target_issues(profile: DatasetProfile) -> list[DiagnosticIssue]:
    """Detect class imbalance and regression target anomalies."""
    if profile.target is None:
        return []

    issues: list[DiagnosticIssue] = []
    target = profile.target

    if target.class_percentages:
        minority_pct = min(target.class_percentages.values())
        if target.is_imbalanced:
            issues.append(
                DiagnosticIssue(
                    id="class_imbalance",
                    title="Class imbalance detected",
                    category="target_analysis",
                    severity=Severity.HIGH if minority_pct < 10 else Severity.MEDIUM,
                    evidence=f"Minority class = {minority_pct:.1f}%",
                    explanation="Accuracy may hide poor minority-class recall on imbalanced targets.",
                    recommendation="Use stratified validation and inspect F1, precision, recall, and PR-AUC.",
                    confidence=0.92,
                )
            )

    if target.skew is not None and abs(target.skew) >= 1.0:
        issues.append(
            DiagnosticIssue(
                id="target_skew",
                title="Skewed regression target",
                category="target_analysis",
                severity=Severity.MEDIUM,
                evidence=f"Target skew = {target.skew:.2f}",
                explanation="Skewed targets can hurt linear models and some error metrics.",
                recommendation="Inspect outliers and consider log or robust transformations.",
                confidence=0.8,
            )
        )

    if target.outlier_count and target.outlier_count > 0:
        issues.append(
            DiagnosticIssue(
                id="target_outliers",
                title="Target outliers detected",
                category="target_analysis",
                severity=Severity.MEDIUM,
                evidence=f"{target.outlier_count} target outliers detected via IQR rule.",
                explanation="Outliers can disproportionately affect regression metrics.",
                recommendation="Review extreme target values and consider robust loss or clipping.",
                confidence=0.78,
            )
        )

    return issues
