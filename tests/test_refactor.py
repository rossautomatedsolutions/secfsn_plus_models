from __future__ import annotations

import pandas as pd
import pytest

from secfsn_plus_models.dataset import build_model_dataset
from secfsn_plus_models.evaluation import decile_spread, evaluate_cross_sectional_model
from secfsn_plus_models.labels import compute_next_period_return
from secfsn_plus_models.metrics import information_coefficient, rank_information_coefficient
from secfsn_plus_models.models.linear import LinearRegressionModel
from secfsn_plus_models.models.tree import RandomForestModel
from secfsn_plus_models.splits import time_based_split
from secfsn_plus_models.trainer import train_and_predict


@pytest.fixture
def sample_features() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [
            ("A", "2020Q1"),
            ("A", "2020Q2"),
            ("A", "2020Q3"),
            ("B", "2020Q1"),
            ("B", "2020Q2"),
            ("B", "2020Q3"),
        ],
        names=["cik", "period"],
    )
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
            "feature_2": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "price": [100.0, 105.0, 110.0, 200.0, 210.0, 220.0],
            "sector": [1, 1, 1, 2, 2, 2],
        },
        index=index,
    )


def test_compute_next_period_return_preserves_multiindex(sample_features: pd.DataFrame) -> None:
    labels = compute_next_period_return(sample_features, "price")

    expected_index = pd.MultiIndex.from_tuples(
        [("A", "2020Q1"), ("A", "2020Q2"), ("B", "2020Q1"), ("B", "2020Q2")],
        names=["cik", "period"],
    )
    assert labels.index.equals(expected_index)
    assert labels.name == "next_period_return"


def test_build_model_dataset_returns_aligned_objects_without_mutation(sample_features: pd.DataFrame) -> None:
    original = sample_features.copy(deep=True)
    labels = compute_next_period_return(sample_features, "price")

    X, y, meta = build_model_dataset(sample_features, labels, exclude_columns=["sector"])

    assert sample_features.equals(original)
    assert X.index.equals(y.index)
    assert meta.index.equals(y.index)
    assert list(meta.columns) == ["sector"]
    assert "sector" not in X.columns


def test_time_based_split_raises_on_empty_test(sample_features: pd.DataFrame) -> None:
    labels = compute_next_period_return(sample_features, "price")
    X, y, _ = build_model_dataset(sample_features, labels, exclude_columns=["sector"])

    with pytest.raises(ValueError, match="test split is empty"):
        time_based_split(X, y, train_end_period="2020Q2", val_end_period="2020Q3")


def test_time_based_split_returns_expected_boundaries(sample_features: pd.DataFrame) -> None:
    labels = compute_next_period_return(sample_features, "price")
    X, y, _ = build_model_dataset(sample_features, labels, exclude_columns=["sector"])
    split = time_based_split(X, y, train_end_period="2020Q1")
    X_train, y_train, X_val, y_val, X_test, y_test = split

    assert X_train.index.equals(y_train.index)
    assert X_val.empty and y_val.empty
    assert X_test.index.equals(y_test.index)
    assert set(X_train.index.get_level_values("period")) == {"2020Q1"}
    assert set(X_test.index.get_level_values("period")) == {"2020Q2"}


def test_metrics_return_period_indexed_series() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("A", "2020Q1"),
            ("B", "2020Q1"),
            ("C", "2020Q1"),
            ("A", "2020Q2"),
            ("B", "2020Q2"),
            ("C", "2020Q2"),
        ],
        names=["cik", "period"],
    )
    y_true = pd.Series([0.1, 0.2, 0.3, 0.3, 0.2, 0.1], index=index)
    y_pred = pd.Series([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], index=index)

    ic = information_coefficient(y_true, y_pred)
    rank_ic = rank_information_coefficient(y_true, y_pred)

    assert list(ic.index) == ["2020Q1", "2020Q2"]
    assert list(rank_ic.index) == ["2020Q1", "2020Q2"]
    assert ic.name == "information_coefficient"
    assert rank_ic.name == "rank_information_coefficient"


def test_evaluation_summary_is_deterministic() -> None:
    rows = []
    for period in ["2020Q1", "2020Q2"]:
        for i in range(10):
            rows.append((f"C{i}", period))
    index = pd.MultiIndex.from_tuples(rows, names=["cik", "period"])
    y_true = pd.Series([float(i) for _period in ["2020Q1", "2020Q2"] for i in range(10)], index=index)
    y_pred = pd.Series([float(i) for _period in ["2020Q1", "2020Q2"] for i in range(10)], index=index)

    spread = decile_spread(y_true, y_pred)
    summary = evaluate_cross_sectional_model(y_true, y_pred)

    assert list(spread.index) == ["2020Q1", "2020Q2"]
    assert summary.index.tolist() == ["IC", "RankIC", "DecileSpread"]
    assert summary.loc["IC"] == pytest.approx(1.0)
    assert summary.loc["RankIC"] == pytest.approx(1.0)
    assert summary.loc["DecileSpread"] == pytest.approx(9.0)


def test_train_and_predict_preserves_index_and_is_deterministic(sample_features: pd.DataFrame) -> None:
    labels = compute_next_period_return(sample_features, "price")
    X, y, _ = build_model_dataset(sample_features, labels, exclude_columns=["sector"])
    X_train, y_train, _, _, X_test, _ = time_based_split(X, y, train_end_period="2020Q1")

    linear_predictions = train_and_predict(LinearRegressionModel(), X_train, y_train, X_test)
    rf_model_a = RandomForestModel(n_estimators=10)
    rf_model_b = RandomForestModel(n_estimators=10)
    rf_pred_a = train_and_predict(rf_model_a, X_train, y_train, X_test)
    rf_pred_b = train_and_predict(rf_model_b, X_train, y_train, X_test)

    assert linear_predictions.index.equals(X_test.index)
    pd.testing.assert_series_equal(rf_pred_a, rf_pred_b)


def test_compute_next_period_return_raises_when_price_column_missing(sample_features: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="missing required columns"):
        compute_next_period_return(sample_features.drop(columns=["price"]), "price")
