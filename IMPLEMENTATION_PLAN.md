# ML Debugger — Implementation Plan

This document translates the PRD into a phased delivery plan and repository layout.

## Repository Structure

```text
ml-debugger/
├── app/
│   ├── app.py                 # Streamlit entrypoint and upload workflow
│   ├── pages/                 # Overview, Data Health, Leakage, Features, Models, Diagnostics, Recommendations
│   └── components/            # Shared session helpers and widgets
├── src/
│   ├── ingestion.py           # CSV/XLSX loading and validation
│   ├── profiling.py           # Schema profiling, target analysis, health score
│   ├── quality.py             # Missing values, duplicates, IDs, cardinality
│   ├── leakage.py             # Possible leakage heuristics
│   ├── feature_analysis.py    # Feature summaries and target relationships
│   ├── preprocessing.py       # ColumnTransformer + train/val/test splits
│   ├── modeling.py            # Baseline sklearn pipelines
│   ├── evaluation.py          # Metrics, over/underfitting, shift
│   ├── explainability.py      # Feature importance extraction
│   ├── diagnostics.py         # Unified diagnostic issue assembly
│   ├── recommendations.py     # Prioritized next steps
│   ├── export.py              # JSON/HTML report export
│   ├── pipeline.py            # End-to-end orchestration
│   ├── schemas.py             # Dataclasses for typed results
│   └── constants.py           # Shared enums and keyword lists
├── tests/
├── examples/
├── reports/
├── requirements.txt
├── README.md
└── IMPLEMENTATION_PLAN.md
```

## Architecture

```text
Upload -> Ingestion -> Profiling -> Quality + Leakage
      -> Preprocessing + Split -> Baseline Training -> Evaluation
      -> Diagnostics -> Recommendations -> Streamlit UI / Export
```

## Phase Breakdown

### Phase 1 — Dataset ingestion and profiling
- Load CSV/XLSX with validation
- Infer schema and column types
- Compute missing values, duplicates, constants, likely IDs
- Analyze target distribution
- Compute overall health score
- Deliverable: `ingestion.py`, `profiling.py`, sample dataset, tests

### Phase 2 — Data quality and leakage diagnostics
- Feature-level quality rules
- Class imbalance and regression target checks
- Name-based and association-based leakage heuristics
- Deliverable: `quality.py`, `leakage.py`, diagnostic schemas

### Phase 3 — Baseline ML models and evaluation
- Leakage-safe preprocessing inside sklearn `Pipeline`
- Dummy, linear/logistic, random forest, gradient boosting baselines
- Classification and regression metrics
- Train/validation comparison, distribution shift checks
- Deliverable: `preprocessing.py`, `modeling.py`, `evaluation.py`, `explainability.py`

### Phase 4 — Diagnostic engine and recommendations
- Normalize all findings into structured issues
- Severity assignment and confidence scoring
- Evidence-backed recommendation ranking
- Deliverable: `diagnostics.py`, `recommendations.py`, `pipeline.py`, export helpers

### Phase 5 — Streamlit UI
- Upload flow, target selection, analyze action
- Pages for overview, health, leakage, features, models, diagnostics, recommendations
- JSON/HTML export from sidebar
- Deliverable: `app/` package

## Design Principles
- Separate detection, evidence, hypothesis, and recommendation
- Use "possible leakage" language unless evidence is conclusive
- Fit preprocessing only inside training pipelines
- Prefer F1/PR-AUC over accuracy for imbalanced classification
- Keep MVP modular so FastAPI or async training can be added later

## Testing Strategy
- Unit tests for ingestion and profiling
- Integration test for full `run_analysis`
- Manual Streamlit verification with `examples/customer_churn.csv`

## Future Work (Post-MVP)
- SHAP explanations
- PDF export
- Hyperparameter search
- FastAPI service layer
- Experiment tracking and dataset versioning
