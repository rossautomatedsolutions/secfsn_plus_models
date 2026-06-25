# secfsn_plus_models

`secfsn_plus_models` is **Layer 04** in the SECFSN stack.

Canonical stack order:

1. `secfsn`
2. `secfsn_plus`
3. `secfsn_plus_structural`
4. `secfsn_plus_models`
5. `secfsn_plus_decisionsystems`

## What This Repository Does

This repository consumes prepared feature matrices from Layer 02 / Layer 03 and provides
the core modeling loop for SECFSN research workflows:

- feature-matrix loading and validation
- forward-label construction
- aligned dataset assembly
- time-based train/test splitting
- deterministic model training and prediction
- evaluation metrics for cross-sectional research
- feature-importance and robustness analysis
- packaging model outputs for downstream Layer 05 consumption

This repository does **not**:

- parse raw SEC filings
- download or wrangle SEC source datasets
- derive canonical accounting ratios
- generate structural diagnostics
- make final long/short/risk decisions

Layer 05 (`secfsn_plus_decisionsystems`) may optionally consume outputs produced here,
especially a canonical `model_risk_probability` column aligned to `(cik, period)`.

## Architecture Position

- **Layer 02 - `secfsn_plus`** prepares canonical feature matrices.
- **Layer 03 - `secfsn_plus_structural`** adds deterministic structural diagnostics.
- **Layer 04 - `secfsn_plus_models`** trains models, evaluates predictions, and packages outputs.
- **Layer 05 - `secfsn_plus_decisionsystems`** consumes prepared deterministic features plus optional model outputs.

## Canonical Input Contract

The canonical modeling input is a `pandas.DataFrame` with:

- index: `MultiIndex(["cik", "period"])`
- canonical period labels: `YYYYQ#`
- numeric feature columns used as model inputs
- optional metadata columns such as `price`, `sector_code`, or provenance fields

Expected upstream behavior:

- the feature matrix is already prepared by Layer 02 / 03
- rows represent company-period observations
- no raw SEC tables or filing-level inputs are passed into this repository

Current label helpers assume:

- a price-like column such as `price` or `adjusted_close`
- one or more future periods exist within each `cik`

## Canonical Output Contract

The core training path returns aligned predictions indexed by `("cik", "period")`.

For downstream handoff, the canonical packaged output is a `pandas.DataFrame` with the same
MultiIndex and at least:

- `model_prediction`
- `model_expected_return`
- `model_risk_probability`

`model_risk_probability` is currently packaged as a deterministic cross-sectional percentile
transform of the prediction series. This is a **Layer 04 output adapter**, not a claim of
probabilistic model calibration.

Additional downstream-facing fields may be added later when warranted, such as:

- `model_confidence`
- `model_expected_drawdown`

## Public API

Primary package entrypoints:

- `load_feature_matrix(...)`
- `build_model_dataset(...)`
- `compute_next_period_return(...)`
- `time_based_split(...)`
- `train_and_predict(...)`
- `evaluate_cross_sectional_model(...)`
- `build_model_output_frame(...)`
- `ModelRegistry`

## Repository Layout

```text
secfsn_plus_models/
├── notebooks/
│   ├── 01_dataset_inspection.ipynb
│   ├── 02_baseline_model_training.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_prediction_evaluation.ipynb
├── examples/
│   └── layer04_smoke.py
├── secfsn_plus_models/
│   ├── automation.py
│   ├── data_loader.py
│   ├── dataset.py
│   ├── evaluation.py
│   ├── labels.py
│   ├── metrics.py
│   ├── outputs.py
│   ├── splits.py
│   ├── trainer.py
│   └── models/
├── tests/
└── README.md
```

## Simple Workflow

```python
from secfsn_plus_models.automation import generate_synthetic_feature_matrix
from secfsn_plus_models.dataset import build_model_dataset
from secfsn_plus_models.labels import compute_next_period_return
from secfsn_plus_models.models.linear import LinearRegressionModel
from secfsn_plus_models.outputs import build_model_output_frame
from secfsn_plus_models.splits import time_based_split
from secfsn_plus_models.trainer import train_and_predict

features_df = generate_synthetic_feature_matrix(price_column="price", seed=20240323)
labels = compute_next_period_return(features_df, "price")
X, y, _ = build_model_dataset(features_df, labels, exclude_columns=["price", "sector_code"])
X_train, y_train, _, _, X_test, _ = time_based_split(X, y, train_end_period="2020Q4")

predictions = train_and_predict(LinearRegressionModel(), X_train, y_train, X_test)
model_outputs = build_model_output_frame(predictions, higher_prediction_more_risky=False)
```

The resulting `model_outputs` frame is shaped for Layer 05 consumption because it preserves
the canonical `(cik, period)` index and includes `model_risk_probability`.

## Examples And Notebooks

- `examples/layer04_smoke.py` provides a deterministic smoke run using synthetic inputs only.
- the notebooks are repository demos for inspection, baseline training, comparison, and evaluation

## Validation

Recommended validation commands:

```powershell
pytest -q
python examples\layer04_smoke.py
```
