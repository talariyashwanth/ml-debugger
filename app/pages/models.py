"""Models page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.widgets import render_model_table
from src.schemas import AnalysisResult


def render(result: AnalysisResult) -> None:
    st.header("Model Performance")
    render_model_table(result)

    best = next((m for m in result.models if m.name == result.best_model_name), None)
    if best and best.train_metrics:
        primary = "r2" if result.problem_type.value == "regression" else "f1"
        st.subheader("Train vs Validation")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Split": "Train",
                        primary.upper(): best.train_metrics.get(primary),
                    },
                    {
                        "Split": "Validation",
                        primary.upper(): best.metrics.get(primary),
                    },
                ]
            ),
            use_container_width=True,
        )

    if best and best.confusion_matrix:
        st.subheader("Confusion Matrix")
        st.dataframe(pd.DataFrame(best.confusion_matrix), use_container_width=True)

    if best and best.feature_importance:
        st.subheader("Feature Importance")
        st.bar_chart(best.feature_importance)
