"""Dataset assembly utilities for model-ready feature and label inputs."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .utils import copy_aligned_frame_and_series


def build_model_dataset(
    features_df: pd.DataFrame,
    label_series: pd.Series,
    exclude_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Assemble aligned model features, labels, and metadata.

    Assumptions
    -----------
    * ``features_df`` and ``label_series`` are indexed by ``MultiIndex(['cik', 'period'])``.
    * Feature columns listed in ``exclude_columns`` should be preserved in ``meta`` and
      excluded from the returned model feature matrix.
    * Inputs are never mutated.

    Input schema
    ------------
    ``features_df``
        DataFrame indexed by ``(cik, period)`` containing numeric model inputs and optional
        metadata columns.
    ``label_series``
        Series indexed by ``(cik, period)`` containing the training label.
    ``exclude_columns``
        Optional iterable of feature columns to move into ``meta``.

    Output schema
    -------------
    Returns ``(X, y, meta)`` where each object preserves the aligned ``(cik, period)``
    MultiIndex. ``X`` excludes the requested metadata columns, ``y`` contains aligned labels,
    and ``meta`` contains only the excluded columns present in ``features_df``.

    Raises
    ------
    ValueError
        If the inputs do not share any common rows or if all feature columns are excluded.
    """
    aligned_features, aligned_labels = copy_aligned_frame_and_series(
        features_df,
        label_series,
        object_names=("features_df", "label_series"),
    )
    excluded = list(exclude_columns or [])

    meta_columns = [column for column in excluded if column in aligned_features.columns]
    meta = aligned_features.loc[:, meta_columns].copy()
    X = aligned_features.drop(columns=meta_columns, errors="ignore")

    if X.empty or len(X.columns) == 0:
        raise ValueError("No feature columns remain after applying exclude_columns.")

    return X, aligned_labels, meta
