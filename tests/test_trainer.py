from __future__ import annotations

from secfsn_plus_models.dataset import build_model_dataset
from secfsn_plus_models.labels import compute_next_period_return
from secfsn_plus_models.models.linear import LinearRegressionModel
from secfsn_plus_models.splits import time_based_split
from secfsn_plus_models.trainer import train_and_predict


def test_train_and_predict_fits_and_returns_aligned_predictions(synthetic_panel) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")
    X, y, _ = build_model_dataset(synthetic_panel, labels, exclude_columns=["price", "sector_code"])
    X_train, y_train, _, _, X_test, _ = time_based_split(X, y, train_end_period="2020Q4")

    predictions = train_and_predict(LinearRegressionModel(), X_train, y_train, X_test)

    assert predictions.index.equals(X_test.index)
    assert predictions.name == "prediction"
    assert predictions.shape == (len(X_test),)
    assert len(predictions) > 0
