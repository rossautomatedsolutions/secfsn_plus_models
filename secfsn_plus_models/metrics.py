"""Cross-sectional performance metrics for quarterly factor models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_and_align(y_true: pd.Series, y_pred: pd.Series) -> tuple[pd.Series, pd.Series]:
    common_index = y_true.index.intersection(y_pred.index)
    true_aligned = y_true.loc[common_index]
    pred_aligned = y_pred.loc[common_index]
    return true_aligned, pred_aligned


def _mean_cross_sectional_correlation(
    y_true: pd.Series,
    y_pred: pd.Series,
    method: str,
) -> float:
    true_aligned, pred_aligned = _validate_and_align(y_true, y_pred)

    frame = pd.DataFrame({"y_true": true_aligned, "y_pred": pred_aligned}).dropna()
    if frame.empty:
        return float("nan")

    per_period = (
        frame.groupby(level="period")
        .apply(lambda g: g["y_true"].corr(g["y_pred"], method=method))
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if per_period.empty:
        return float("nan")
    return float(per_period.mean())


def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Average per-period Pearson information coefficient (IC)."""
    return _mean_cross_sectional_correlation(y_true=y_true, y_pred=y_pred, method="pearson")


def rank_information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Average per-period Spearman rank information coefficient (Rank IC)."""
    return _mean_cross_sectional_correlation(y_true=y_true, y_pred=y_pred, method="spearman")
