"""Time-based splitting utilities for quarterly model training workflows."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


def _period_to_ordinal(periods: pd.Index) -> pd.Series:
    """Convert canonical period labels ``YYYYQ#`` into sortable ordinals."""
    period_series = pd.Series(periods, index=periods)
    extracted = period_series.str.extract(r"^(?P<year>\d{4})Q(?P<quarter>[1-4])$")
    if extracted.isna().any().any():
        raise ValueError("All period labels must follow canonical format 'YYYYQ#'.")

    years = extracted["year"].astype(int)
    quarters = extracted["quarter"].astype(int)
    return years * 4 + (quarters - 1)


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_end_period: str,
    val_end_period: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split dataset by canonical quarterly period boundaries.

    Parameters
    ----------
    X
        Feature dataframe indexed by ``(cik, period)``.
    y
        Label series indexed by ``(cik, period)``.
    train_end_period
        Inclusive end period for training set.
    val_end_period
        Inclusive end period for validation set. If omitted, validation is
        returned empty and all periods after ``train_end_period`` are test.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        ``X_train, y_train, X_val, y_val, X_test, y_test``.
    """
    common_index = X.index.intersection(y.index)
    X_aligned = X.loc[common_index]
    y_aligned = y.loc[common_index]

    periods = X_aligned.index.get_level_values("period")
    period_ord = _period_to_ordinal(pd.Index(periods))
    train_end_ord = _period_to_ordinal(pd.Index([train_end_period])).iloc[0]

    train_mask = period_ord <= train_end_ord

    if val_end_period is None:
        val_mask = pd.Series(False, index=period_ord.index)
        test_mask = period_ord > train_end_ord
    else:
        val_end_ord = _period_to_ordinal(pd.Index([val_end_period])).iloc[0]
        if val_end_ord < train_end_ord:
            raise ValueError("val_end_period must be >= train_end_period.")
        val_mask = (period_ord > train_end_ord) & (period_ord <= val_end_ord)
        test_mask = period_ord > val_end_ord

    return (
        X_aligned.loc[train_mask.values],
        y_aligned.loc[train_mask.values],
        X_aligned.loc[val_mask.values],
        y_aligned.loc[val_mask.values],
        X_aligned.loc[test_mask.values],
        y_aligned.loc[test_mask.values],
    )
