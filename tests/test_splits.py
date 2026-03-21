from __future__ import annotations

import pandas as pd
import pytest

from secfsn_plus_models.dataset import build_model_dataset
from secfsn_plus_models.labels import compute_next_period_return
from secfsn_plus_models.splits import time_based_split


def test_time_based_split_has_correct_boundaries_and_no_overlap(synthetic_panel: pd.DataFrame) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")
    X, y, _ = build_model_dataset(synthetic_panel, labels, exclude_columns=["price"])

    X_train, y_train, X_val, y_val, X_test, y_test = time_based_split(
        X,
        y,
        train_end_period="2020Q3",
        val_end_period="2021Q1",
    )

    assert set(X_train.index.get_level_values("period")) == {"2020Q1", "2020Q2", "2020Q3"}
    assert set(X_val.index.get_level_values("period")) == {"2020Q4", "2021Q1"}
    assert set(X_test.index.get_level_values("period")) == {"2021Q2"}
    assert X_train.index.equals(y_train.index)
    assert X_val.index.equals(y_val.index)
    assert X_test.index.equals(y_test.index)
    assert set(X_train.index).isdisjoint(set(X_val.index))
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert set(X_val.index).isdisjoint(set(X_test.index))


def test_time_based_split_preserves_chronological_order_within_splits(synthetic_panel: pd.DataFrame) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")
    X, y, _ = build_model_dataset(synthetic_panel, labels, exclude_columns=["price"])

    X_train, _, X_val, _, X_test, _ = time_based_split(
        X,
        y,
        train_end_period="2020Q2",
        val_end_period="2021Q1",
    )

    for split_frame in (X_train, X_val, X_test):
        period_ordinals = split_frame.index.get_level_values("period").map(
            lambda value: int(value[:4]) * 4 + int(value[-1])
        )
        assert list(period_ordinals) == sorted(period_ordinals)


def test_time_based_split_rejects_invalid_period_boundary_order(synthetic_panel: pd.DataFrame) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")
    X, y, _ = build_model_dataset(synthetic_panel, labels, exclude_columns=["price"])

    with pytest.raises(ValueError, match="greater than or equal to train_end_period"):
        time_based_split(X, y, train_end_period="2021Q1", val_end_period="2020Q4")
