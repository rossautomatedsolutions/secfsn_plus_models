"""Shared validation and alignment helpers for SECFSN+ modeling modules."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

INDEX_NAMES = ["cik", "period"]


def validate_multiindex_pandas_object(obj: pd.DataFrame | pd.Series, object_name: str) -> None:
    """Validate that a pandas object is indexed by ``(cik, period)``.

    Parameters
    ----------
    obj
        Pandas dataframe or series expected to use a ``MultiIndex``.
    object_name
        Human-readable name used in error messages.

    Raises
    ------
    TypeError
        If ``obj`` is not a pandas dataframe or series.
    ValueError
        If ``obj`` does not use a MultiIndex containing ``cik`` and ``period``.
    """
    if not isinstance(obj, (pd.DataFrame, pd.Series)):
        raise TypeError(f"{object_name} must be a pandas DataFrame or Series.")

    if not isinstance(obj.index, pd.MultiIndex):
        raise ValueError(f"{object_name} must be indexed by MultiIndex(['cik', 'period']).")

    if list(obj.index.names) != INDEX_NAMES:
        raise ValueError(
            f"{object_name} index names must be exactly {INDEX_NAMES}; got {list(obj.index.names)}."
        )


def copy_aligned_frame_and_series(
    features_df: pd.DataFrame,
    label_series: pd.Series,
    *,
    object_names: tuple[str, str] = ("features_df", "label_series"),
    require_non_empty: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return copies of aligned pandas objects sharing the intersection index.

    Parameters
    ----------
    features_df
        Feature dataframe indexed by ``(cik, period)``.
    label_series
        Label series indexed by ``(cik, period)``.
    object_names
        Names used in validation and error reporting.
    require_non_empty
        Whether to raise if no common index rows exist.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Copies of the aligned dataframe and series using the shared index.

    Raises
    ------
    ValueError
        If the objects do not share any rows.
    """
    frame_name, series_name = object_names
    validate_multiindex_pandas_object(features_df, frame_name)
    validate_multiindex_pandas_object(label_series, series_name)

    common_index = features_df.index[features_df.index.isin(label_series.index)]
    if require_non_empty and common_index.empty:
        raise ValueError(f"{frame_name} and {series_name} do not share any common (cik, period) rows.")

    return features_df.loc[common_index].copy(), label_series.loc[common_index].copy()


def copy_aligned_series_pair(
    left: pd.Series,
    right: pd.Series,
    *,
    object_names: tuple[str, str] = ("left", "right"),
    require_non_empty: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Return copies of aligned series sharing the intersection index."""
    left_name, right_name = object_names
    validate_multiindex_pandas_object(left, left_name)
    validate_multiindex_pandas_object(right, right_name)

    common_index = left.index[left.index.isin(right.index)]
    if require_non_empty and common_index.empty:
        raise ValueError(f"{left_name} and {right_name} do not share any common (cik, period) rows.")

    return left.loc[common_index].copy(), right.loc[common_index].copy()


def validate_required_columns(features_df: pd.DataFrame, required_columns: Iterable[str], object_name: str) -> None:
    """Validate that ``features_df`` contains every required column."""
    missing_columns = [column for column in required_columns if column not in features_df.columns]
    if missing_columns:
        raise KeyError(f"{object_name} is missing required columns: {missing_columns}.")
