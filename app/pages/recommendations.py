"""Recommendations page."""

from __future__ import annotations

import streamlit as st

from src.schemas import AnalysisResult


def render(result: AnalysisResult) -> None:
    st.header("Recommended Next Steps")
    if not result.recommendations:
        st.success("No recommendations generated.")
        return

    for index, rec in enumerate(result.recommendations, start=1):
        st.write(f"**{index:02d}** {rec.recommendation} — **{rec.severity.value.upper()}**")
        st.caption(rec.evidence)
