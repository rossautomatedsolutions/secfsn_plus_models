from __future__ import annotations

import pandas as pd
import pytest

from secfsn_plus_models.metrics import information_coefficient, rank_information_coefficient


def test_information_coefficient_is_correct_for_simple_case(
    manual_metric_panel: tuple[pd.Series, pd.Series],
) -> None:
    y_true, y_pred = manual_metric_panel

    ic = information_coefficient(y_true, y_pred)

    expected = pd.Series(
        [1.0, 1.0],
        index=pd.Index(["2020Q1", "2020Q2"], name="period"),
        name="information_coefficient",
    )
    pd.testing.assert_series_equal(ic, expected)


def test_rank_information_coefficient_respects_ranking(
    manual_metric_panel: tuple[pd.Series, pd.Series],
) -> None:
    y_true, _ = manual_metric_panel
    reversed_pred = pd.Series(
        [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
        index=y_true.index,
        name="prediction",
    )

    rank_ic = rank_information_coefficient(y_true, reversed_pred)

    assert rank_ic.loc["2020Q1"] == pytest.approx(-1.0)
    assert rank_ic.loc["2020Q2"] == pytest.approx(-1.0)


def test_constant_predictions_drop_nan_correlations() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            ("0001", "2020Q1"),
            ("0002", "2020Q1"),
            ("0003", "2020Q1"),
        ],
        names=["cik", "period"],
    )
    y_true = pd.Series([0.1, 0.2, 0.3], index=index)
    constant_pred = pd.Series([1.0, 1.0, 1.0], index=index)

    ic = information_coefficient(y_true, constant_pred)
    rank_ic = rank_information_coefficient(y_true, constant_pred)

    assert ic.empty
    assert rank_ic.empty
