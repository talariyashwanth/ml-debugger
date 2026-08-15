"""Tests for ML Debugger."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.export import export_json_report
from src.ingestion import IngestionError, load_dataset
from src.pipeline import profile_only, run_analysis
from src.profiling import infer_problem_type, profile_dataset


EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "customer_churn.csv"


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_data():
    if not EXAMPLE_PATH.exists():
        from examples.generate_sample import build_sample_classification

        build_sample_classification(EXAMPLE_PATH)


def test_load_dataset():
    df = load_dataset(EXAMPLE_PATH)
    assert len(df) == 500
    assert "churn" in df.columns


def test_invalid_dataset_raises():
    with pytest.raises(IngestionError):
        load_dataset("missing.csv")


def test_profile_dataset():
    df = load_dataset(EXAMPLE_PATH)
    profile = profile_dataset(df, target_column="churn", dataset_name="customer_churn.csv")
    assert profile.row_count == 500
    assert profile.feature_count == 7
    assert 0 <= profile.health_score <= 100
    assert profile.target is not None
    assert profile.target.is_imbalanced


def test_infer_problem_type():
    df = load_dataset(EXAMPLE_PATH)
    assert infer_problem_type(df["churn"]).value == "binary_classification"


def test_profile_only_helper():
    profile = profile_only(EXAMPLE_PATH, target_column="churn")
    assert profile.dataset_name == "customer_churn.csv"


def test_full_analysis_pipeline():
    result = run_analysis(EXAMPLE_PATH, "churn")
    assert result.best_model_name is not None
    assert len(result.models) >= 4
    assert len(result.diagnostics) > 0
    assert len(result.recommendations) > 0
    assert any(issue.category == "leakage" for issue in result.diagnostics)


def test_export_json(tmp_path):
    result = run_analysis(EXAMPLE_PATH, "churn")
    output = export_json_report(result, tmp_path / "report.json")
    assert output.exists()
    assert output.read_text(encoding="utf-8").startswith("{")
