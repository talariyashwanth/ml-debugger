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
streamlit run app/app.py
```

## Usage
1. Upload a CSV or XLSX file in the Streamlit app.
2. Select the target column.
3. Click **Analyze Dataset**.
4. Review pages for health, leakage, models, diagnostics, and recommendations.
5. Export a JSON or HTML report from the sidebar.

## Project Docs
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)

## Sample Dataset
`examples/customer_churn.csv` includes realistic churn signals plus a intentionally suspicious `customer_status` feature for leakage testing.
