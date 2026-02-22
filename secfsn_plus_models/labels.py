"""Label construction utilities for quarterly cross-sectional modeling."""

from __future__ import annotations

import pandas as pd


INDEX_NAMES = ["cik", "period"]


def compute_next_period_return(
    features_df: pd.DataFrame,
    price_column: str,
) -> pd.Series:
    """Compute next-period returns aligned to the current period.

    For each ``cik`` this computes:

    .. math::
        r_t = \frac{P_{t+1}}{P_t} - 1

    and places ``r_t`` at index ``(cik, period_t)``. The final available period
    for each ``cik`` is dropped since no forward return exists.

    Parameters
    ----------
    features_df
        Input feature table indexed by ``MultiIndex(['cik', 'period'])``.
    price_column
        Name of the numeric price column used to calculate returns.

    Returns
    -------
    pd.Series
        Forward return label indexed by ``(cik, period)``.

    Raises
    ------
    KeyError
        If ``price_column`` is not present in ``features_df``.
    ValueError
        If index is not a MultiIndex containing ``cik`` and ``period``.
    """
    if price_column not in features_df.columns:
        raise KeyError(f"Column '{price_column}' not found in features_df.")

    if not isinstance(features_df.index, pd.MultiIndex):
        raise ValueError("features_df must be indexed by MultiIndex(['cik', 'period']).")

    if not set(INDEX_NAMES).issubset(features_df.index.names):
        raise ValueError("features_df index must include levels ['cik', 'period'].")

    price_series = features_df[price_column]
    next_price = price_series.groupby(level="cik", sort=False).shift(-1)
    forward_return = (next_price / price_series) - 1.0
    forward_return = forward_return.dropna()
    forward_return.name = "next_period_return"

    return forward_return
