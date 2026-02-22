"""Dataset assembly utilities for model-ready feature/label matrices."""

from __future__ import annotations

from typing import List, Optional, Tuple

import pandas as pd


def build_model_dataset(
    features_df: pd.DataFrame,
    label_series: pd.Series,
    exclude_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build aligned modeling inputs from feature and label data.

    Parameters
    ----------
    features_df
        Feature table indexed by ``(cik, period)``.
    label_series
        Label series indexed by ``(cik, period)``.
    exclude_columns
        Optional list of columns to exclude from model features. These columns
        are returned in ``meta`` when present.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        ``X`` model features, ``y`` aligned labels, and ``meta`` dataframe
        containing excluded columns.
    """
    exclude_columns = exclude_columns or []

    common_index = features_df.index.intersection(label_series.index)
    aligned_features = features_df.loc[common_index]
    y = label_series.loc[common_index]

    meta_columns = [col for col in exclude_columns if col in aligned_features.columns]
    meta = aligned_features.loc[:, meta_columns].copy()

    X = aligned_features.drop(columns=meta_columns, errors="ignore")

    return X, y, meta
