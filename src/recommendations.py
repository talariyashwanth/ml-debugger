"""Recommendation engine."""

from __future__ import annotations

from src.constants import SEVERITY_ORDER
from src.schemas import AnalysisResult, DiagnosticIssue


PRIORITY_CATEGORIES = [
    "data_quality",
    "leakage",
    "target_analysis",
    "evaluation",
    "feature_quality",
    "generalization",
    "distribution_shift",
    "modeling",
]


def generate_recommendations(diagnostics: list[DiagnosticIssue]) -> list[DiagnosticIssue]:
    """Convert diagnostics into prioritized, evidence-backed recommendations."""
    recommendations: list[DiagnosticIssue] = []

    for issue in sorted(
        diagnostics,
        key=lambda item: (
            SEVERITY_ORDER[item.severity],
            PRIORITY_CATEGORIES.index(item.category)
            if item.category in PRIORITY_CATEGORIES
            else len(PRIORITY_CATEGORIES),
            item.title,
        ),
    ):
        recommendations.append(
            DiagnosticIssue(
                id=f"rec_{issue.id}",
                title=issue.title,
                category=issue.category,
                severity=issue.severity,
                evidence=issue.evidence,
                explanation=issue.explanation,
                recommendation=issue.recommendation,
                confidence=issue.confidence,
                feature=issue.feature,
            )
        )

    return recommendations


def summarize_recommendations(result: AnalysisResult) -> dict[str, int]:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for issue in result.diagnostics:
        counts[issue.severity.value] += 1
    return counts
