"""Pure metric computation functions for cross-sectional evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import copy_aligned_series_pair


def _cross_sectional_correlation_by_period(
    y_true: pd.Series,
    y_pred: pd.Series,
    method: str,
) -> pd.Series:
    """Compute per-period cross-sectional correlations for aligned predictions.

    Assumptions
    -----------
    * ``y_true`` and ``y_pred`` are indexed by ``MultiIndex(['cik', 'period'])``.
    * Correlation is computed independently within each period.

    Input schema
    ------------
    ``y_true`` and ``y_pred``
        Series indexed by ``(cik, period)``.
    ``method``
        Correlation method accepted by :meth:`pandas.Series.corr`.

    Output schema
    -------------
    Series indexed by ``period`` containing one correlation value per period.
    """
    true_aligned, pred_aligned = copy_aligned_series_pair(
        y_true,
        y_pred,
        object_names=("y_true", "y_pred"),
        require_non_empty=False,
    )

    frame = pd.DataFrame({"y_true": true_aligned, "y_pred": pred_aligned}).dropna()
    if frame.empty:
        return pd.Series(dtype=float, name=f"{method}_correlation")

    correlations = frame.groupby(level="period", sort=False).apply(
        lambda group: group["y_true"].corr(group["y_pred"], method=method)
    )
    correlations = correlations.replace([np.inf, -np.inf], np.nan).dropna()
    correlations.name = f"{method}_correlation"
    correlations.index.name = "period"
    return correlations


def information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Compute period-level Pearson information coefficients.

    Assumptions
    -----------
    ``y_true`` and ``y_pred`` use aligned ``(cik, period)`` MultiIndex rows.

    Input schema
    ------------
    ``y_true`` and ``y_pred`` are series indexed by ``(cik, period)``.

    Output schema
    -------------
    Series indexed by ``period`` containing the Pearson information coefficient for each period.
    """
    ic = _cross_sectional_correlation_by_period(y_true=y_true, y_pred=y_pred, method="pearson")
    ic.name = "information_coefficient"
    return ic


def rank_information_coefficient(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Compute period-level Spearman rank information coefficients.

    Assumptions
    -----------
    ``y_true`` and ``y_pred`` use aligned ``(cik, period)`` MultiIndex rows.

    Input schema
    ------------
    ``y_true`` and ``y_pred`` are series indexed by ``(cik, period)``.

    Output schema
    -------------
    Series indexed by ``period`` containing the Spearman rank information coefficient for each period.
    """
    rank_ic = _cross_sectional_correlation_by_period(
        y_true=y_true,
        y_pred=y_pred,
        method="spearman",
    )
    rank_ic.name = "rank_information_coefficient"
    return rank_ic
