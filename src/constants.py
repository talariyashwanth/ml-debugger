"""Shared constants for ML Debugger."""

from enum import Enum


class ProblemType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


SEVERITY_ORDER = {
    Severity.CRITICAL: 0,
    Severity.HIGH: 1,
    Severity.MEDIUM: 2,
    Severity.LOW: 3,
}

LEAKAGE_KEYWORDS = [
    "target",
    "label",
    "outcome",
    "churn",
    "default",
    "fraud",
    "status",
    "result",
    "prediction",
    "leak",
]

ID_KEYWORDS = ["id", "uuid", "guid", "key", "index", "identifier", "customer_id", "user_id"]
