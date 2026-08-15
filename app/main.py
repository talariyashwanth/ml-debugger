"""ML Debugger Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.session import get_analysis_result, set_analysis_result
from app.pages import (
    data_health,
    diagnostics,
    features,
    leakage,
    models,
    overview,
    recommendations,
)
from src.export import export_html_report, export_json_report
from src.ingestion import load_dataset
from src.pipeline import run_analysis
from src.profiling import infer_problem_type

st.set_page_config(page_title="ML Debugger", page_icon="🩺", layout="wide")

PAGES = {
    "Overview": overview.render,
    "Data Health": data_health.render,
    "Leakage": leakage.render,
    "Features": features.render,
    "Models": models.render,
    "Diagnostics": diagnostics.render,
    "Recommendations": recommendations.render,
}


def main() -> None:
    st.title("ML Debugger")
    st.caption("Diagnose your ML pipeline before you ship.")

    uploaded = st.file_uploader("Drop CSV or XLSX here", type=["csv", "xlsx", "xls"])
    df = None
    if uploaded is not None:
        temp_path = Path("reports") / uploaded.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(uploaded.getbuffer())
        df = load_dataset(temp_path)
        st.session_state["dataset_path"] = str(temp_path)
        st.session_state["dataframe"] = df

    df = st.session_state.get("dataframe")
    if df is None:
        st.info("Upload a dataset to begin.")
        return

    st.write(f"Dataset loaded with **{len(df):,}** rows and **{df.shape[1]}** columns.")
    target_column = st.selectbox("Target column", options=list(df.columns))
    problem_type = infer_problem_type(df[target_column])
    st.write(f"Detected problem type: **{problem_type.value.replace('_', ' ').title()}**")

    if st.button("Analyze Dataset", type="primary"):
        with st.spinner("Running diagnostics and baseline models..."):
            result = run_analysis(st.session_state["dataset_path"], target_column)
            set_analysis_result(result)
        st.success("Analysis complete.")

    result = get_analysis_result()
    if result is None:
        return

    page = st.sidebar.radio("Navigation", list(PAGES.keys()))
    PAGES[page](result)

    st.sidebar.divider()
    st.sidebar.subheader("Export")
    if st.sidebar.button("Export JSON report"):
        output = export_json_report(result, Path("reports") / f"{result.dataset_name}.json")
        st.sidebar.success(f"Saved to {output}")
    if st.sidebar.button("Export HTML report"):
        output = export_html_report(result, Path("reports") / f"{result.dataset_name}.html")
        st.sidebar.success(f"Saved to {output}")


if __name__ == "__main__":
    main()
