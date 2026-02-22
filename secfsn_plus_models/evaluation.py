"""Evaluation routines for quarterly cross-sectional model outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import information_coefficient, rank_information_coefficient


def _period_decile_spread(group: pd.DataFrame) -> float:
    """Compute top-minus-bottom decile spread for one period."""
    if group["y_pred"].nunique() < 10:
        return float("nan")

    deciles = pd.qcut(group["y_pred"], q=10, labels=False, duplicates="drop")
    if deciles.isna().all():
        return float("nan")

    valid = group.assign(decile=deciles).dropna(subset=["decile", "y_true"])
    if valid.empty:
        return float("nan")

    if valid["decile"].nunique() != 10:
        return float("nan")

    mean_returns = valid.groupby("decile")["y_true"].mean()
    if not set(range(10)).issubset({int(decile) for decile in mean_returns.index}):
        return float("nan")

    return float(mean_returns.loc[9] - mean_returns.loc[0])


def evaluate_cross_sectional_model(
    y_true: pd.Series,
    y_pred: pd.Series,
    periods: pd.Index,
) -> dict[str, float]:
    """Evaluate model predictions with IC, Rank IC, and decile spread.

    Parameters
    ----------
    y_true
        Realized next-period returns indexed by ``(cik, period)``.
    y_pred
        Model predictions indexed by ``(cik, period)``.
    periods
        Period labels to include in evaluation.

    Returns
    -------
    dict[str, float]
        Dictionary containing ``IC``, ``RankIC``, and ``DecileSpread``.
    """
    mask_true = y_true.index.get_level_values("period").isin(periods)
    mask_pred = y_pred.index.get_level_values("period").isin(periods)

    y_true_f = y_true.loc[mask_true]
    y_pred_f = y_pred.loc[mask_pred]

    common_index = y_true_f.index.intersection(y_pred_f.index)
    frame = pd.DataFrame(
        {
            "y_true": y_true_f.loc[common_index],
            "y_pred": y_pred_f.loc[common_index],
        }
    ).dropna()

    if frame.empty:
        return {"IC": float("nan"), "RankIC": float("nan"), "DecileSpread": float("nan")}

    spread_by_period = frame.groupby(level="period").apply(_period_decile_spread)
    spread = float(spread_by_period.replace([np.inf, -np.inf], np.nan).dropna().mean())

    return {
        "IC": information_coefficient(frame["y_true"], frame["y_pred"]),
        "RankIC": rank_information_coefficient(frame["y_true"], frame["y_pred"]),
        "DecileSpread": spread,
    }
