"""Feature-level analysis helpers."""

from __future__ import annotations

import pandas as pd
from scipy.stats import skew


def analyze_features(df: pd.DataFrame, target_column: str) -> list[dict]:
    """Summarize feature distributions and target relationships."""
    summaries: list[dict] = []
    target = df[target_column]

    for column in df.columns:
        if column == target_column:
            continue
        series = df[column]
        summary = {
            "feature": column,
            "dtype": str(series.dtype),
            "missing_pct": round(series.isna().mean() * 100, 2),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            summary.update(
                {
                    "mean": float(numeric.mean()) if not numeric.empty else None,
                    "std": float(numeric.std(ddof=0)) if not numeric.empty else None,
                    "skew": float(skew(numeric)) if len(numeric) > 2 else None,
                }
            )
        else:
            rates = (
                df.groupby(series.astype(str))[target_column]
                .mean(numeric_only=True)
                .to_dict()
                if pd.api.types.is_numeric_dtype(target)
                else df.groupby(series.astype(str))[target_column].apply(lambda s: s.mode().iloc[0]).to_dict()
            )
            summary["category_target_summary"] = {str(k): v for k, v in list(rates.items())[:5]}

        summaries.append(summary)

    return summaries
