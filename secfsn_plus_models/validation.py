"""Centralized structural validation helpers for SECFSN+ modeling inputs."""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from pandas.api.types import is_numeric_dtype


INDEX_NAMES = ["cik", "period"]
PERIOD_PATTERN = re.compile(r"^(?P<year>\d{4})Q(?P<quarter>[1-4])$")


def validate_multiindex_pandas_object(
    obj: pd.DataFrame | pd.Series,
    object_name: str,
) -> None:
    """Require a pandas object indexed by canonical ``(cik, period)`` pairs."""

    if not isinstance(obj, (pd.DataFrame, pd.Series)):
        raise TypeError(f"{object_name} must be a pandas DataFrame or Series.")

    if not isinstance(obj.index, pd.MultiIndex):
        raise ValueError(f"{object_name} must be indexed by MultiIndex(['cik', 'period']).")

    if list(obj.index.names) != INDEX_NAMES:
        raise ValueError(
            f"{object_name} index names must be {INDEX_NAMES}, got {list(obj.index.names)}."
        )


def validate_required_columns(
    features_df: pd.DataFrame,
    required_columns: Iterable[str],
    object_name: str,
) -> None:
    missing_columns = [column for column in required_columns if column not in features_df.columns]
    if missing_columns:
        raise ValueError(f"{object_name} is missing required columns: {missing_columns}.")


def validate_unique_index_rows(
    obj: pd.DataFrame | pd.Series,
    object_name: str,
) -> None:
    validate_multiindex_pandas_object(obj, object_name)
    if obj.index.has_duplicates:
        duplicate_rows = obj.index[obj.index.duplicated()].tolist()
        raise ValueError(f"{object_name} contains duplicate index rows: {duplicate_rows[:5]}.")


def period_to_ordinal(
    periods: pd.Index,
    *,
    object_name: str = "period",
) -> pd.Series:
    """Convert canonical quarterly labels like ``2025Q3`` into sortable ordinals."""

    normalized = periods.astype(str)
    extracted = normalized.to_series(index=range(len(normalized))).str.extract(PERIOD_PATTERN)
    invalid_mask = extracted.isna().any(axis=1)
    if invalid_mask.any():
        bad_periods = normalized[invalid_mask.to_numpy()].tolist()
        raise ValueError(
            f"{object_name} must use canonical 'YYYYQ#' labels; invalid values include {bad_periods[:5]}."
        )

    year = extracted["year"].astype(int)
    quarter = extracted["quarter"].astype(int)
    return ((year * 4) + (quarter - 1)).rename("period_ordinal")


def validate_canonical_period_index(
    obj: pd.DataFrame | pd.Series,
    object_name: str,
) -> None:
    validate_multiindex_pandas_object(obj, object_name)
    period_to_ordinal(obj.index.get_level_values("period"), object_name=f"{object_name} period")


def validate_sorted_period_index(
    obj: pd.DataFrame | pd.Series,
    object_name: str,
) -> None:
    validate_multiindex_pandas_object(obj, object_name)
    ordinal = period_to_ordinal(obj.index.get_level_values("period"), object_name=f"{object_name} period")
    expected = pd.DataFrame(
        {
            "cik": obj.index.get_level_values("cik").astype(str),
            "period": obj.index.get_level_values("period").astype(str),
            "period_ordinal": ordinal.to_numpy(),
        }
    ).sort_values(["cik", "period_ordinal"], kind="mergesort")

    actual_pairs = list(zip(obj.index.get_level_values("cik"), obj.index.get_level_values("period")))
    expected_pairs = list(zip(expected["cik"], expected["period"]))
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"{object_name} must be sorted chronologically by MultiIndex(['cik', 'period']) before modeling."
        )


def validate_numeric_column(
    features_df: pd.DataFrame,
    column_name: str,
    object_name: str,
) -> None:
    validate_required_columns(features_df, [column_name], object_name)
    if not is_numeric_dtype(features_df[column_name]):
        raise ValueError(f"{object_name}.{column_name} must be numeric.")


def validate_numeric_modeling_columns(
    features_df: pd.DataFrame,
    object_name: str,
) -> None:
    for column_name in features_df.columns:
        if not is_numeric_dtype(features_df[column_name]):
            raise ValueError(f"{object_name}.{column_name} must be numeric.")
