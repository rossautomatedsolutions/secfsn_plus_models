"""Time-based dataset splitting utilities."""

from __future__ import annotations

import pandas as pd

from .utils import copy_aligned_frame_and_series


_PERIOD_PATTERN = r"^(?P<year>\d{4})Q(?P<quarter>[1-4])$"


def _period_to_ordinal(periods: pd.Index) -> pd.Series:
    """Convert canonical ``YYYYQ#`` labels into sortable ordinals."""
    period_series = pd.Series(periods.astype(str), index=periods)
    extracted = period_series.str.extract(_PERIOD_PATTERN)
    if extracted.isna().any().any():
        raise ValueError("All period labels must follow canonical format 'YYYYQ#'.")

    years = extracted["year"].astype(int)
    quarters = extracted["quarter"].astype(int)
    return years * 4 + (quarters - 1)


def _validate_non_empty_split(name: str, X_split: pd.DataFrame, y_split: pd.Series, *, required: bool) -> None:
    """Validate that a split is non-empty when required."""
    if required and (X_split.empty or y_split.empty):
        raise ValueError(f"The {name} split is empty. Adjust the requested period boundaries.")


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_end_period: str,
    val_end_period: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split aligned features and labels by inclusive quarterly period cutoffs.

    Assumptions
    -----------
    * ``X`` and ``y`` are indexed by ``MultiIndex(['cik', 'period'])``.
    * Period labels follow the canonical ``YYYYQ#`` format.
    * Input ordering is preserved within each returned split.

    Input schema
    ------------
    ``X``
        Feature dataframe indexed by ``(cik, period)``.
    ``y``
        Label series indexed by ``(cik, period)``.
    ``train_end_period``
        Inclusive upper bound of the training split.
    ``val_end_period``
        Optional inclusive upper bound of the validation split.

    Output schema
    -------------
    Returns ``(X_train, y_train, X_val, y_val, X_test, y_test)``. Every object preserves the
    aligned ``(cik, period)`` MultiIndex and contains a subset of the input rows.

    Raises
    ------
    ValueError
        If period labels are invalid, split boundaries are inconsistent, or a required split is empty.
    """
    X_aligned, y_aligned = copy_aligned_frame_and_series(X, y, object_names=("X", "y"))

    periods = pd.Index(X_aligned.index.get_level_values("period"))
    period_ord = pd.Series(_period_to_ordinal(periods).to_numpy(), index=X_aligned.index)
    train_end_ord = int(_period_to_ordinal(pd.Index([train_end_period])).iloc[0])

    train_mask = period_ord <= train_end_ord

    if val_end_period is None:
        val_mask = pd.Series(False, index=X_aligned.index)
        test_mask = period_ord > train_end_ord
        validation_required = False
    else:
        val_end_ord = int(_period_to_ordinal(pd.Index([val_end_period])).iloc[0])
        if val_end_ord < train_end_ord:
            raise ValueError("val_end_period must be greater than or equal to train_end_period.")
        val_mask = (period_ord > train_end_ord) & (period_ord <= val_end_ord)
        test_mask = period_ord > val_end_ord
        validation_required = True

    X_train, y_train = X_aligned.loc[train_mask].copy(), y_aligned.loc[train_mask].copy()
    X_val, y_val = X_aligned.loc[val_mask].copy(), y_aligned.loc[val_mask].copy()
    X_test, y_test = X_aligned.loc[test_mask].copy(), y_aligned.loc[test_mask].copy()

    X_train = X_train.sort_index(level=["period", "cik"])
    y_train = y_train.sort_index(level=["period", "cik"])
    X_val = X_val.sort_index(level=["period", "cik"])
    y_val = y_val.sort_index(level=["period", "cik"])
    X_test = X_test.sort_index(level=["period", "cik"])
    y_test = y_test.sort_index(level=["period", "cik"])

    _validate_non_empty_split("train", X_train, y_train, required=True)
    _validate_non_empty_split("validation", X_val, y_val, required=validation_required)
    _validate_non_empty_split("test", X_test, y_test, required=True)

    return X_train, y_train, X_val, y_val, X_test, y_test
