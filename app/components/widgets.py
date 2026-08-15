"""Reusable Streamlit UI components."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.schemas import AnalysisResult, DiagnosticIssue


def severity_badge(severity: str) -> str:
    mapping = {
        "critical": "🔴 CRITICAL",
        "high": "🟠 HIGH",
        "medium": "🟡 MEDIUM",
        "low": "🔵 LOW",
    }
    return mapping.get(severity, severity.upper())


def render_issue_list(issues: list[DiagnosticIssue]) -> None:
    if not issues:
        st.success("No issues detected in this section.")
        return

    for issue in issues:
        with st.expander(f"{severity_badge(issue.severity.value)} — {issue.title}"):
            st.write(f"**Evidence:** {issue.evidence}")
            st.write(f"**Why it matters:** {issue.explanation}")
            st.write(f"**Recommendation:** {issue.recommendation}")


def render_model_table(result: AnalysisResult) -> None:
    rows = []
    primary = "r2" if result.problem_type.value == "regression" else "f1"
    for model in result.models:
        row = {"Model": model.name, primary.upper(): model.metrics.get(primary)}
        if "roc_auc" in model.metrics:
            row["ROC-AUC"] = model.metrics["roc_auc"]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
