"""Real feature-matrix loading helpers for SECFSN+ model inputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .validation import (
    period_to_ordinal,
    validate_canonical_period_index,
    validate_multiindex_pandas_object,
    validate_sorted_period_index,
    validate_unique_index_rows,
)


SUPPORTED_FILE_SUFFIXES = {".csv", ".parquet"}


def _normalize_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a loaded feature matrix onto canonical ``(cik, period)`` indexing."""

    if {"cik", "period"}.issubset(frame.columns):
        normalized = frame.assign(
            cik=frame["cik"].astype("string").str.strip(),
            period=frame["period"].astype("string").str.strip().str.upper(),
        ).set_index(["cik", "period"])
    elif isinstance(frame.index, pd.MultiIndex):
        normalized = frame.copy()
        normalized.index = normalized.index.set_names(["cik", "period"])
        index_frame = normalized.index.to_frame(index=False).assign(
            cik=lambda df: df["cik"].astype("string").str.strip(),
            period=lambda df: df["period"].astype("string").str.strip().str.upper(),
        )
        normalized.index = pd.MultiIndex.from_frame(index_frame[["cik", "period"]])
    else:
        raise ValueError(
            "Loaded feature matrix must either already use MultiIndex(['cik', 'period']) "
            "or contain cik/period columns."
        )

    validate_multiindex_pandas_object(normalized, "features_df")

    index_frame = normalized.index.to_frame(index=False).copy()
    period_ord = period_to_ordinal(
        pd.Index(index_frame["period"]),
        object_name="features_df period",
    ).to_numpy()
    ordered_rows = index_frame.assign(period_ordinal=period_ord).sort_values(
        ["cik", "period_ordinal"],
        kind="mergesort",
    )
    sorted_index = pd.MultiIndex.from_frame(ordered_rows[["cik", "period"]])
    sorted_frame = normalized.loc[sorted_index]

    validate_unique_index_rows(sorted_frame, "features_df")
    validate_canonical_period_index(sorted_frame, "features_df")
    validate_sorted_period_index(sorted_frame, "features_df")
    return sorted_frame


def load_feature_matrix(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet feature matrix into the canonical modeling layout."""

    candidate = Path(path).expanduser()
    suffix = candidate.suffix.lower()
    if suffix not in SUPPORTED_FILE_SUFFIXES:
        raise ValueError(
            f"Unsupported feature matrix format '{suffix}'. Expected one of {sorted(SUPPORTED_FILE_SUFFIXES)}."
        )

    if suffix == ".parquet":
        frame = pd.read_parquet(candidate)
    else:
        frame = pd.read_csv(candidate)

    return _normalize_feature_matrix(frame)
