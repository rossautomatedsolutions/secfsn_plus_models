"""Evaluation routines built from pure metric computations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import information_coefficient, rank_information_coefficient
from .utils import copy_aligned_series_pair


def decile_spread(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Compute period-level top-minus-bottom prediction decile spread.

    Assumptions
    -----------
    * ``y_true`` and ``y_pred`` are indexed by ``MultiIndex(['cik', 'period'])``.
    * Deciles are formed independently within each period using prediction ranks.

    Input schema
    ------------
    ``y_true`` and ``y_pred``
        Series indexed by ``(cik, period)``.

    Output schema
    -------------
    Series indexed by ``period`` containing the mean realized return of the top prediction decile
    minus the bottom prediction decile for each eligible period.
    """
    true_aligned, pred_aligned = copy_aligned_series_pair(
        y_true,
        y_pred,
        object_names=("y_true", "y_pred"),
        require_non_empty=False,
    )
    frame = pd.DataFrame({"y_true": true_aligned, "y_pred": pred_aligned}).dropna()
    if frame.empty:
        return pd.Series(dtype=float, name="decile_spread")

    def _spread_for_period(group: pd.DataFrame) -> float:
        if group["y_pred"].nunique() < 10:
            return float("nan")

        deciles = pd.qcut(group["y_pred"], q=10, labels=False, duplicates="drop")
        if deciles.isna().all():
            return float("nan")

        valid_group = group.assign(decile=deciles).dropna(subset=["decile", "y_true"])
        if valid_group["decile"].nunique() != 10:
            return float("nan")

        mean_returns = valid_group.groupby("decile")["y_true"].mean()
        return float(mean_returns.loc[9] - mean_returns.loc[0])

    spreads = frame.groupby(level="period", sort=False).apply(_spread_for_period)
    spreads = spreads.replace([np.inf, -np.inf], np.nan).dropna()
    spreads.name = "decile_spread"
    spreads.index.name = "period"
    return spreads


def decile_return_summary(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Compute global top/bottom prediction decile returns across aligned rows.

    Assumptions
    -----------
    * ``y_true`` and ``y_pred`` are indexed by ``MultiIndex(['cik', 'period'])``.
    * Deciles are formed across the full aligned evaluation sample using prediction ranks.

    Input schema
    ------------
    ``y_true`` and ``y_pred``
        Series indexed by ``(cik, period)``.

    Output schema
    -------------
    Series indexed by ``['top_decile_return', 'bottom_decile_return', 'spread']``.
    """
    true_aligned, pred_aligned = copy_aligned_series_pair(
        y_true,
        y_pred,
        object_names=("y_true", "y_pred"),
        require_non_empty=False,
    )
    frame = pd.DataFrame({"y_true": true_aligned, "y_pred": pred_aligned}).dropna()
    if frame.empty:
        return pd.Series(
            {
                "top_decile_return": float("nan"),
                "bottom_decile_return": float("nan"),
                "spread": float("nan"),
            },
            dtype=float,
            name="decile_return_summary",
        )

    ranked_predictions = frame["y_pred"].rank(method="first")
    deciles = pd.qcut(ranked_predictions, q=10, labels=False, duplicates="drop")
    valid_frame = frame.assign(decile=deciles).dropna(subset=["decile"])
    if valid_frame.empty:
        return pd.Series(
            {
                "top_decile_return": float("nan"),
                "bottom_decile_return": float("nan"),
                "spread": float("nan"),
            },
            dtype=float,
            name="decile_return_summary",
        )

    mean_returns = valid_frame.groupby("decile", sort=True)["y_true"].mean()
    top_decile_return = float(mean_returns.iloc[-1])
    bottom_decile_return = float(mean_returns.iloc[0])
    summary = pd.Series(
        {
            "top_decile_return": top_decile_return,
            "bottom_decile_return": bottom_decile_return,
            "spread": top_decile_return - bottom_decile_return,
        },
        dtype=float,
        name="decile_return_summary",
    )
    return summary


def evaluate_cross_sectional_model(
    y_true: pd.Series,
    y_pred: pd.Series,
    periods: pd.Index | None = None,
) -> pd.Series:
    """Summarize IC, Rank IC, and decile spread for selected periods.

    Assumptions
    -----------
    * ``y_true`` and ``y_pred`` are indexed by ``MultiIndex(['cik', 'period'])``.
    * If ``periods`` is provided, it contains period labels present in the input index.

    Input schema
    ------------
    ``y_true`` and ``y_pred``
        Series indexed by ``(cik, period)``.
    ``periods``
        Optional index of period labels to include.

    Output schema
    -------------
    Series indexed by ``['IC', 'RankIC', 'DecileSpread']`` containing the mean value of each
    metric across the selected periods.
    """
    true_aligned, pred_aligned = copy_aligned_series_pair(
        y_true,
        y_pred,
        object_names=("y_true", "y_pred"),
        require_non_empty=False,
    )

    if periods is not None:
        period_filter = pd.Index(periods)
        true_mask = true_aligned.index.get_level_values("period").isin(period_filter)
        pred_mask = pred_aligned.index.get_level_values("period").isin(period_filter)
        true_aligned = true_aligned.loc[true_mask].copy()
        pred_aligned = pred_aligned.loc[pred_mask].copy()

    if true_aligned.empty or pred_aligned.empty or true_aligned.index.intersection(pred_aligned.index).empty:
        return pd.Series(
            {"IC": float("nan"), "RankIC": float("nan"), "DecileSpread": float("nan")},
            dtype=float,
            name="evaluation_summary",
        )

    ic = information_coefficient(true_aligned, pred_aligned)
    rank_ic = rank_information_coefficient(true_aligned, pred_aligned)
    spread = decile_spread(true_aligned, pred_aligned)

    summary = pd.Series(
        {
            "IC": float(ic.mean()) if not ic.empty else float("nan"),
            "RankIC": float(rank_ic.mean()) if not rank_ic.empty else float("nan"),
            "DecileSpread": float(spread.mean()) if not spread.empty else float("nan"),
        },
        dtype=float,
    )
    summary.name = "evaluation_summary"
    return summary
