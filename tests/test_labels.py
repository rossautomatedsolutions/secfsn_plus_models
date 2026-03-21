from __future__ import annotations

import pandas as pd
import pytest

from secfsn_plus_models.labels import compute_next_period_return


def test_next_period_return_is_correctly_aligned(synthetic_panel: pd.DataFrame) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")

    expected = (
        synthetic_panel.groupby(level="cik", sort=False)["price"].shift(-1) / synthetic_panel["price"]
    ) - 1.0
    expected = expected.dropna().rename("next_period_return")

    pd.testing.assert_series_equal(labels, expected)


def test_last_period_is_dropped_for_every_company(synthetic_panel: pd.DataFrame) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")
    last_period = synthetic_panel.index.get_level_values("period").unique()[-1]

    assert last_period not in labels.index.get_level_values("period")
    assert len(labels) == len(synthetic_panel) - synthetic_panel.index.get_level_values("cik").nunique()


def test_label_construction_does_not_leak_beyond_the_next_period(synthetic_panel: pd.DataFrame) -> None:
    original = compute_next_period_return(synthetic_panel, "price")
    modified = synthetic_panel.copy()
    modified.loc[("0001", "2021Q3"), "price"] = 10_000.0

    updated = compute_next_period_return(modified, "price")

    changed_index = pd.MultiIndex.from_tuples([("0001", "2021Q2")], names=["cik", "period"])
    unchanged = original.index.difference(changed_index)

    pd.testing.assert_series_equal(original.loc[unchanged], updated.loc[unchanged])
    assert updated.loc[("0001", "2021Q2")] != original.loc[("0001", "2021Q2")]


def test_missing_price_column_raises_key_error(synthetic_panel: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="missing required columns"):
        compute_next_period_return(synthetic_panel.drop(columns=["price"]), "price")


def test_missing_next_period_raises_value_error(single_period_panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no rows have a next period"):
        compute_next_period_return(single_period_panel, "price")
