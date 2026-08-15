"""Leakage page."""

from __future__ import annotations

import streamlit as st

from src.schemas import AnalysisResult


def render(result: AnalysisResult) -> None:
    st.header("Possible Leakage")
    if not result.leakage_findings:
        st.success("No possible leakage flags detected.")
        return

    for finding in result.leakage_findings:
        st.error(f"POSSIBLE LEAKAGE — {finding.feature}")
        st.write(f"**Risk:** {finding.severity.value.upper()}")
        if finding.correlation is not None:
            st.write(f"**Association:** {finding.correlation:.2f}")
        st.write(f"**Evidence:** {finding.evidence}")
        st.write(f"**Why suspicious:** {finding.explanation}")
        st.write(f"**Recommended action:** {finding.recommendation}")
