"""Diagnostics page."""

from __future__ import annotations

import streamlit as st

from app.components.widgets import render_issue_list
from src.schemas import AnalysisResult


def render(result: AnalysisResult) -> None:
    st.header("Diagnostics")
    grouped = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": [],
    }
    for issue in result.diagnostics:
        grouped[issue.severity.value].append(issue)

    for severity, issues in grouped.items():
        if not issues:
            continue
        st.subheader(severity.title())
        render_issue_list(issues)
