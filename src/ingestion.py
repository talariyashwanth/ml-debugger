"""Dataset ingestion and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class IngestionError(ValueError):
    """Raised when dataset ingestion fails."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel dataset into a pandas DataFrame."""
    file_path = Path(path)
    if not file_path.exists():
        raise IngestionError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if suffix == ".csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
    except Exception as exc:
        raise IngestionError(f"Failed to read dataset: {exc}") from exc

    return validate_dataframe(df)


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an uploaded dataframe."""
    if df is None or df.empty:
        raise IngestionError("Dataset is empty.")

    if df.shape[1] < 2:
        raise IngestionError("Dataset must contain at least one feature and one target column.")

    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    if cleaned.columns.duplicated().any():
        raise IngestionError("Dataset contains duplicate column names.")

    return cleaned


def get_dataset_summary(df: pd.DataFrame) -> dict[str, int]:
    """Return basic dataset dimensions."""
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
    }
