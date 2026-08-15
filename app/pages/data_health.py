"""Data health page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.widgets import render_issue_list
from src.schemas import AnalysisResult


def render(result: AnalysisResult) -> None:
    st.header("Data Health")

    rows = [
        {
            "Feature": col.name,
            "Type": "numeric" if "int" in col.dtype or "float" in col.dtype else "category",
            "Missing": f"{col.missing_pct:.1f}%",
            "Unique": col.unique_count,
            "Risk": col.risk,
        }
        for col in result.profile.columns
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if result.profile.target and result.profile.target.class_percentages:
        st.subheader("Target Distribution")
        st.bar_chart(result.profile.target.class_percentages)

    st.subheader("Quality Issues")
    render_issue_list(result.quality_issues)
