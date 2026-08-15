"""Overview page."""

from __future__ import annotations

import streamlit as st

from src.recommendations import summarize_recommendations
from src.schemas import AnalysisResult


def render(result: AnalysisResult) -> None:
    st.header("Overview")
    counts = summarize_recommendations(result)
    primary = "r2" if result.problem_type.value == "regression" else "f1"
    best = next((m for m in result.models if m.name == result.best_model_name), None)

    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Health", f"{result.profile.health_score}/100")
    col2.metric("Critical Issues", counts["critical"])
    col3.metric("High Issues", counts["high"])

    st.subheader("Dataset Summary")
    st.write(
        {
            "Rows": result.profile.row_count,
            "Features": result.profile.feature_count,
            "Missing values": f"{result.profile.missing_pct:.1f}%",
            "Duplicates": f"{result.profile.duplicate_pct:.1f}%",
        }
    )

    if best:
        st.subheader("Best Baseline")
        st.write(f"**{best.name}** — {primary.upper()}: {best.metrics.get(primary, 0):.2f}")

    if result.diagnostics:
        st.subheader("Top Issues")
        for issue in result.diagnostics[:5]:
            st.warning(f"{issue.severity.value.upper()}: {issue.title} — {issue.evidence}")
