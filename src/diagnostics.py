"""Diagnostic engine."""

from __future__ import annotations

from src.constants import SEVERITY_ORDER, Severity
from src.schemas import AnalysisResult, DiagnosticIssue, LeakageFinding, ModelMetrics
from src.quality import detect_quality_issues, detect_target_issues


def leakage_to_diagnostics(findings: list[LeakageFinding]) -> list[DiagnosticIssue]:
    diagnostics: list[DiagnosticIssue] = []
    for finding in findings:
        diagnostics.append(
            DiagnosticIssue(
                id=f"leakage_{finding.feature}",
                title=f"Possible data leakage: {finding.feature}",
                category="leakage",
                severity=finding.severity,
                evidence=finding.evidence,
                explanation=finding.explanation,
                recommendation=finding.recommendation,
                confidence=0.8 if finding.severity == Severity.CRITICAL else 0.7,
                feature=finding.feature,
            )
        )
    return diagnostics


def model_comparison_diagnostics(
    models: list[ModelMetrics],
    best_model_name: str | None,
    problem_type,
) -> list[DiagnosticIssue]:
    if not models or not best_model_name:
        return []

    metric = "r2" if problem_type.value == "regression" else "f1"
    sorted_models = sorted(models, key=lambda m: m.metrics.get(metric, float("-inf")), reverse=(metric == "r2"))
    best = next(m for m in sorted_models if m.name == best_model_name)
    dummy = next((m for m in sorted_models if m.name == "Dummy"), None)
    diagnostics: list[DiagnosticIssue] = []

    if dummy:
        improvement = best.metrics.get(metric, 0) - dummy.metrics.get(metric, 0)
        diagnostics.append(
            DiagnosticIssue(
                id="baseline_improvement",
                title="Baseline model comparison",
                category="modeling",
                severity=Severity.LOW,
                evidence=(
                    f"{best_model_name} validation {metric.upper()} = {best.metrics.get(metric, 0):.2f}; "
                    f"Dummy = {dummy.metrics.get(metric, 0):.2f}"
                ),
                explanation="Comparing against a naive baseline helps validate that learned signal exists.",
                recommendation=(
                    f"Use {best_model_name} as the current baseline and investigate feature quality "
                    "before adding model complexity."
                ),
                confidence=0.85,
            )
        )
        if improvement < 0.05 and metric != "r2":
            diagnostics.append(
                DiagnosticIssue(
                    id="weak_signal",
                    title="Weak predictive signal detected",
                    category="modeling",
                    severity=Severity.HIGH,
                    evidence=f"Best model improves validation {metric.upper()} over Dummy by only {improvement:.2f}.",
                    explanation="The feature set may not contain enough usable signal for the target.",
                    recommendation="Review feature engineering, target definition, and possible leakage or label noise.",
                    confidence=0.74,
                )
            )
    return diagnostics


def build_diagnostics(result: AnalysisResult) -> list[DiagnosticIssue]:
    diagnostics: list[DiagnosticIssue] = []
    diagnostics.extend(result.quality_issues)
    diagnostics.extend(leakage_to_diagnostics(result.leakage_findings))
    diagnostics.extend(result.overfitting_issues)
    diagnostics.extend(result.underfitting_issues)
    diagnostics.extend(result.distribution_shift)
    diagnostics.extend(
        model_comparison_diagnostics(result.models, result.best_model_name, result.problem_type)
    )
    return sorted(
        diagnostics,
        key=lambda issue: (SEVERITY_ORDER[issue.severity], issue.title),
    )
