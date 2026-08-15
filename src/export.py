"""Report export utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.schemas import AnalysisResult


def export_json_report(result: AnalysisResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset_name": result.dataset_name,
        "problem_type": result.problem_type.value,
        "target_column": result.target_column,
        "profile": asdict(result.profile),
        "diagnostics": [asdict(item) for item in result.diagnostics],
        "recommendations": [asdict(item) for item in result.recommendations],
        "models": [
            {
                "name": model.name,
                "metrics": model.metrics,
                "train_metrics": model.train_metrics,
                "confusion_matrix": model.confusion_matrix,
                "feature_importance": model.feature_importance,
            }
            for model in result.models
        ],
        "best_model_name": result.best_model_name,
    }

    for section in ("diagnostics", "recommendations"):
        for item in payload[section]:
            item["severity"] = item["severity"].value if hasattr(item["severity"], "value") else item["severity"]

    profile_target = payload["profile"].get("target")
    if profile_target and profile_target.get("problem_type"):
        profile_target["problem_type"] = profile_target["problem_type"].value

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def export_html_report(result: AnalysisResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        f"<h1>ML Debugger Report: {result.dataset_name}</h1>",
        f"<p><strong>Problem type:</strong> {result.problem_type.value}</p>",
        f"<p><strong>Health score:</strong> {result.profile.health_score}/100</p>",
        "<h2>Diagnostics</h2><ul>",
    ]
    for issue in result.diagnostics:
        sections.append(
            f"<li><strong>{issue.severity.value.upper()}</strong> - {issue.title}: {issue.evidence}</li>"
        )
    sections.append("</ul><h2>Recommendations</h2><ol>")
    for rec in result.recommendations:
        sections.append(f"<li>{rec.recommendation}</li>")
    sections.append("</ol>")

    path.write_text("\n".join(sections), encoding="utf-8")
    return path
