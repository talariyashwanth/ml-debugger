"""Generate example datasets for local testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_sample_classification(path: Path) -> Path:
    rng = np.random.default_rng(42)
    n = 500
    tenure = rng.integers(1, 72, size=n)
    monthly_charges = rng.normal(65, 20, size=n).round(2)
    support_calls = rng.poisson(2, size=n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.5, 0.3, 0.2])
    region = rng.choice(["North", "South", "East", "West"], size=n)
    customer_id = [f"CUST-{i:05d}" for i in range(n)]

    churn_score = (
        0.02 * (72 - tenure)
        + 0.015 * monthly_charges
        + 0.25 * support_calls
        + np.where(contract == "Month-to-month", 1.5, 0.0)
        + rng.normal(0, 0.5, size=n)
    )
    churn = (churn_score > np.quantile(churn_score, 0.85)).astype(int)
    customer_status = np.where(churn == 1, "Churned", "Active")

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "support_calls": support_calls,
            "contract": contract,
            "region": region,
            "customer_status": customer_status,
            "churn": churn,
        }
    )
    df.loc[rng.choice(n, size=20, replace=False), "monthly_charges"] = np.nan
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    build_sample_classification(Path("examples/customer_churn.csv"))
    print("Wrote examples/customer_churn.csv")
