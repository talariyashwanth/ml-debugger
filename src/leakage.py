"""Possible data leakage detection."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, spearmanr

from src.constants import LEAKAGE_KEYWORDS, ProblemType, Severity
from src.schemas import LeakageFinding


def detect_leakage(
    df: pd.DataFrame,
    target_column: str,
    problem_type: ProblemType,
) -> list[LeakageFinding]:
    """Flag features with suspicious target association or naming patterns."""
    findings: list[LeakageFinding] = []
    target = df[target_column]
    feature_columns = [col for col in df.columns if col != target_column]

    for column in feature_columns:
        name_finding = _name_based_leakage(column)
        if name_finding:
            findings.append(name_finding)
            continue

        association = _association_strength(df[column], target, problem_type)
        if association is None:
            continue

        correlation, method = association
        if abs(correlation) >= 0.95:
            findings.append(
                LeakageFinding(
                    feature=column,
                    severity=Severity.CRITICAL,
                    correlation=round(correlation, 4),
                    evidence=f"{method} association with target = {correlation:.2f}",
                    explanation=(
                        f"'{column}' is near-perfectly associated with the target. "
                        "It may contain information available only after the target event."
                    ),
                    recommendation="Investigate feature creation time and exclude if post-outcome.",
                )
            )
        elif abs(correlation) >= 0.85:
            findings.append(
                LeakageFinding(
                    feature=column,
                    severity=Severity.HIGH,
                    correlation=round(correlation, 4),
                    evidence=f"{method} association with target = {correlation:.2f}",
                    explanation=(
                        f"'{column}' has a suspiciously strong relationship with the target."
                    ),
                    recommendation="Review how this feature is generated before using it in modeling.",
                )
            )

    return findings


def _name_based_leakage(column: str) -> LeakageFinding | None:
    lowered = column.lower()
    if any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in LEAKAGE_KEYWORDS):
        return LeakageFinding(
            feature=column,
            severity=Severity.HIGH,
            correlation=None,
            evidence=f"Feature name '{column}' resembles a target or post-outcome field.",
            explanation="The feature name suggests it may encode outcome-related information.",
            recommendation="Verify when this feature becomes available relative to the target event.",
        )
    return None


def _association_strength(
    feature: pd.Series,
    target: pd.Series,
    problem_type: ProblemType,
) -> tuple[float, str] | None:
    aligned = pd.concat([feature, target], axis=1).dropna()
    if aligned.empty:
        return None

    feature = aligned.iloc[:, 0]
    target = aligned.iloc[:, 1]

    if problem_type == ProblemType.REGRESSION and pd.api.types.is_numeric_dtype(feature):
        if feature.nunique(dropna=True) <= 1:
            return None
        corr, _ = pearsonr(feature.astype(float), pd.to_numeric(target, errors="coerce").astype(float))
        return float(corr), "Pearson"

    if pd.api.types.is_numeric_dtype(feature) and problem_type != ProblemType.REGRESSION:
        encoded_target = pd.factorize(target)[0]
        corr, _ = spearmanr(feature.astype(float), encoded_target)
        return float(corr), "Spearman"

    contingency = pd.crosstab(feature.astype(str), target.astype(str))
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None
    chi2, _, _, _ = chi2_contingency(contingency)
    n = contingency.values.sum()
    phi = np.sqrt(chi2 / max(n, 1))
    return float(phi), "Cramér's V"
