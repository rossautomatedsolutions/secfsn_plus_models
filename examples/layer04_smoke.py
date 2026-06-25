"""Deterministic Layer 04 smoke example using synthetic inputs only."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from secfsn_plus_models.automation import generate_synthetic_feature_matrix
from secfsn_plus_models.dataset import build_model_dataset
from secfsn_plus_models.labels import compute_next_period_return
from secfsn_plus_models.models.linear import LinearRegressionModel
from secfsn_plus_models.outputs import build_model_output_frame
from secfsn_plus_models.splits import time_based_split
from secfsn_plus_models.trainer import train_and_predict

DISPLAY_COLUMNS: tuple[str, ...] = (
    "model_prediction",
    "model_expected_return",
    "model_risk_probability",
)


def run_layer04_smoke() -> pd.DataFrame:
    """Train a small deterministic model and package Layer 04 outputs."""
    features_df = generate_synthetic_feature_matrix(price_column="price", seed=20240323)
    labels = compute_next_period_return(features_df, "price")
    X, y, _ = build_model_dataset(features_df, labels, exclude_columns=["price", "sector_code"])
    X_train, y_train, _, _, X_test, _ = time_based_split(X, y, train_end_period="2020Q4")

    predictions = train_and_predict(LinearRegressionModel(), X_train, y_train, X_test)
    model_outputs = build_model_output_frame(
        predictions,
        higher_prediction_more_risky=False,
    )
    return model_outputs


def main() -> None:
    model_outputs = run_layer04_smoke()
    print("Layer 04 smoke example: model outputs compatible with Layer 05 model bridge")
    print(model_outputs.loc[:, list(DISPLAY_COLUMNS)].head(10).round(4).to_string())


if __name__ == "__main__":
    main()
