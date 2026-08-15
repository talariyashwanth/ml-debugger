"""Features page."""

from __future__ import annotations

import streamlit as st

from src.feature_analysis import analyze_features
from src.schemas import AnalysisResult


def render(result: AnalysisResult) -> None:
    st.header("Feature Analysis")
    df = st.session_state.get("dataframe")
    if df is None:
        st.warning("Feature details require the uploaded dataset in session.")
        return

    summaries = analyze_features(df, result.target_column)
    for summary in summaries[:15]:
        with st.expander(summary["feature"]):
            st.json(summary)

    best = next((m for m in result.models if m.name == result.best_model_name), None)
    if best and best.feature_importance:
        st.subheader("Top Features")
        st.bar_chart(best.feature_importance)
