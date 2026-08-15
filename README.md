# ML Debugger

Automated ML diagnostics platform for tabular supervised-learning projects.

## Features
- CSV/XLSX ingestion and dataset profiling
- Data-quality and possible leakage detection
- Reproducible baseline models with sklearn pipelines
- Overfitting, underfitting, and distribution-shift checks
- Evidence-backed diagnostics and recommendations
- Streamlit dashboard with JSON/HTML export

## Quick Start

```powershell
cd C:\Users\talari\Projects\ml-debugger
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python examples/generate_sample.py
pytest
streamlit run app/main.py
```

## Usage
1. Upload a CSV or XLSX file in the Streamlit app.
2. Select the target column.
3. Click **Analyze Dataset**.
4. Review pages for health, leakage, models, diagnostics, and recommendations.
5. Export a JSON or HTML report from the sidebar.

## Sample Datasets

Use these bundled files to quickly test ML Debugger end to end:

| File | Format | Rows | Suggested target | What it exercises |
|------|--------|------|------------------|-------------------|
| [`examples/ml_debugger_test_dataset.csv`](examples/ml_debugger_test_dataset.csv) | CSV | 44 | `churn` or `churn_flag` | Missing values, duplicate rows, likely ID column, possible leakage |
| [`examples/ml_debugger_test_dataset.xlsx`](examples/ml_debugger_test_dataset.xlsx) | XLSX | 44 | `churn` or `churn_flag` | Same dataset as CSV — verifies Excel ingestion |
| [`examples/customer_churn.csv`](examples/customer_churn.csv) | CSV | 500 | `churn` | Larger sample with class imbalance and leakage signals |

**Columns:** `customer_id`, `age`, `region`, `annual_income`, `support_calls`, `churn`, `customer_status`, `churn_flag`, `monthly_logins`

**Expected findings on the test dataset:**
- Duplicate rows (`customer_id` 1003 and 1015 appear twice)
- Missing values in `age` and `annual_income`
- Likely identifier: `customer_id`
- Possible leakage: `customer_status`, `churn_flag`
- Baseline model comparison on classification metrics

### Test in the Streamlit app

```powershell
streamlit run app/main.py
```

1. Upload `examples/ml_debugger_test_dataset.csv` or `examples/ml_debugger_test_dataset.xlsx`
2. Set target to **`churn`**
3. Click **Analyze Dataset**
4. Check **Data Health**, **Leakage**, **Models**, and **Recommendations**

### Test from the command line

```powershell
python -c "from src.pipeline import run_analysis; r = run_analysis('examples/ml_debugger_test_dataset.csv', 'churn'); print(r.profile.health_score, r.best_model_name, len(r.diagnostics))"
```

```powershell
python -c "from src.pipeline import run_analysis; r = run_analysis('examples/ml_debugger_test_dataset.xlsx', 'churn_flag'); print(r.profile.health_score, r.best_model_name, len(r.diagnostics))"
```

## Project Docs
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
