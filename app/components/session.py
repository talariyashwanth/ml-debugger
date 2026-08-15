"""Shared Streamlit session helpers."""

from __future__ import annotations

import streamlit as st

from src.schemas import AnalysisResult


def get_analysis_result() -> AnalysisResult | None:
    return st.session_state.get("analysis_result")


def set_analysis_result(result: AnalysisResult) -> None:
    st.session_state["analysis_result"] = result
