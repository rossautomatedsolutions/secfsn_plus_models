from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from secfsn_plus_models.dataset import build_model_dataset
from secfsn_plus_models.labels import compute_next_period_return
from secfsn_plus_models.models.linear import LinearRegressionModel
from secfsn_plus_models.outputs import build_model_output_frame
from secfsn_plus_models.splits import time_based_split
from secfsn_plus_models.trainer import train_and_predict


def test_build_model_output_frame_preserves_index_and_emits_canonical_columns(synthetic_panel: pd.DataFrame) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")
    X, y, _ = build_model_dataset(synthetic_panel, labels, exclude_columns=["price", "sector_code"])
    X_train, y_train, _, _, X_test, _ = time_based_split(X, y, train_end_period="2020Q4")
    predictions = train_and_predict(LinearRegressionModel(), X_train, y_train, X_test)

    output = build_model_output_frame(predictions, higher_prediction_more_risky=False)

    assert output.index.equals(predictions.index)
    assert output.columns.tolist() == [
        "model_prediction",
        "model_expected_return",
        "model_risk_probability",
    ]
    assert output["model_prediction"].equals(predictions.rename("model_prediction"))
    assert output["model_risk_probability"].between(0.0, 1.0, inclusive="both").all()


def test_build_model_output_frame_is_deterministic_for_fixed_predictions(manual_metric_panel: tuple[pd.Series, pd.Series]) -> None:
    _y_true, y_pred = manual_metric_panel

    first = build_model_output_frame(y_pred, higher_prediction_more_risky=False)
    second = build_model_output_frame(y_pred, higher_prediction_more_risky=False)

    pd.testing.assert_frame_equal(first, second)


def test_layer04_smoke_example_returns_canonical_output_shape() -> None:
    script_path = Path(__file__).resolve().parents[1] / "examples" / "layer04_smoke.py"
    spec = importlib.util.spec_from_file_location("layer04_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    output = module.run_layer04_smoke()

    assert isinstance(output, pd.DataFrame)
    assert list(output.index.names) == ["cik", "period"]
    assert "model_risk_probability" in output.columns
    assert "model_prediction" in output.columns
    assert output["model_risk_probability"].between(0.0, 1.0, inclusive="both").all()
