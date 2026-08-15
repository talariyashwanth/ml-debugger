"""End-to-end analysis orchestration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.diagnostics import build_diagnostics
from src.evaluation import (
    detect_distribution_shift,
    detect_overfitting,
    detect_underfitting,
    evaluate_all_models,
    select_best_model,
)
from src.explainability import attach_importance_to_best_model
from src.ingestion import load_dataset
from src.leakage import detect_leakage
from src.modeling import train_baseline_models
from src.preprocessing import split_dataset
from src.profiling import infer_problem_type, profile_dataset
from src.quality import detect_quality_issues, detect_target_issues
from src.recommendations import generate_recommendations
from src.schemas import AnalysisResult


def run_analysis(
    dataset_path: str | Path,
    target_column: str,
) -> AnalysisResult:
    """Run the full ML Debugger analysis pipeline."""
    path = Path(dataset_path)
    df = load_dataset(path)
    profile = profile_dataset(df, target_column=target_column, dataset_name=path.name)
    problem_type = infer_problem_type(df[target_column])

    quality_issues = detect_quality_issues(profile)
    quality_issues.extend(detect_target_issues(profile))
    leakage_findings = detect_leakage(df, target_column, problem_type)

    splits = split_dataset(df, target_column)
    trained_models = train_baseline_models(splits, df)
    model_metrics = evaluate_all_models(trained_models, splits)
    best_model_name = select_best_model(model_metrics, problem_type)
    attach_importance_to_best_model(trained_models, best_model_name, splits, model_metrics)

    overfitting_issues = []
    underfitting_issues = []
    for model in model_metrics:
        overfit = detect_overfitting(model, problem_type)
        underfit = detect_underfitting(model, problem_type)
        if overfit:
            overfitting_issues.append(overfit)
        if underfit:
            underfitting_issues.append(underfit)

    distribution_shift = detect_distribution_shift(splits, splits["feature_columns"])

    result = AnalysisResult(
        dataset_name=path.name,
        problem_type=problem_type,
        target_column=target_column,
        profile=profile,
        quality_issues=quality_issues,
        leakage_findings=leakage_findings,
        diagnostics=[],
        recommendations=[],
        models=model_metrics,
        best_model_name=best_model_name,
        distribution_shift=distribution_shift,
        overfitting_issues=overfitting_issues,
        underfitting_issues=underfitting_issues,
    )
    result.diagnostics = build_diagnostics(result)
    result.recommendations = generate_recommendations(result.diagnostics)
    return result


def profile_only(dataset_path: str | Path, target_column: str | None = None):
    """Phase 1 helper: load and profile without modeling."""
    path = Path(dataset_path)
    df = load_dataset(path)
    return profile_dataset(df, target_column=target_column, dataset_name=path.name)
