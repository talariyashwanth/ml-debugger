"""Data schemas for ML Debugger results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.constants import ProblemType, Severity


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    missing_pct: float
    unique_count: int
    is_constant: bool
    is_near_constant: bool
    is_likely_id: bool
    risk: str
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class TargetProfile:
    problem_type: ProblemType
    class_counts: dict[str, int] | None = None
    class_percentages: dict[str, float] | None = None
    is_imbalanced: bool = False
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    skew: float | None = None
    outlier_count: int | None = None


@dataclass
class DatasetProfile:
    dataset_name: str
    row_count: int
    column_count: int
    feature_count: int
    missing_pct: float
    duplicate_pct: float
    numeric_feature_count: int
    categorical_feature_count: int
    constant_column_count: int
    likely_id_column_count: int
    health_score: int
    columns: list[ColumnProfile]
    target: TargetProfile | None = None


@dataclass
class DiagnosticIssue:
    id: str
    title: str
    category: str
    severity: Severity
    evidence: str
    explanation: str
    recommendation: str
    confidence: float
    feature: str | None = None


@dataclass
class LeakageFinding:
    feature: str
    severity: Severity
    correlation: float | None
    evidence: str
    explanation: str
    recommendation: str


@dataclass
class ModelMetrics:
    name: str
    metrics: dict[str, float]
    train_metrics: dict[str, float] | None = None
    confusion_matrix: list[list[int]] | None = None
    feature_importance: dict[str, float] | None = None


@dataclass
class AnalysisResult:
    dataset_name: str
    problem_type: ProblemType
    target_column: str
    profile: DatasetProfile
    quality_issues: list[DiagnosticIssue]
    leakage_findings: list[LeakageFinding]
    diagnostics: list[DiagnosticIssue]
    recommendations: list[DiagnosticIssue]
    models: list[ModelMetrics]
    best_model_name: str | None = None
    distribution_shift: list[DiagnosticIssue] = field(default_factory=list)
    overfitting_issues: list[DiagnosticIssue] = field(default_factory=list)
    underfitting_issues: list[DiagnosticIssue] = field(default_factory=list)
