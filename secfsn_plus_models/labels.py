"""Label construction logic for quarterly cross-sectional modeling."""

from __future__ import annotations

import pandas as pd

from .utils import validate_multiindex_pandas_object, validate_required_columns


def compute_next_period_return(
    features_df: pd.DataFrame,
    price_column: str,
) -> pd.Series:
    """Compute current-period aligned forward returns from a price column.

    Assumptions
    -----------
    * ``features_df`` is indexed by ``MultiIndex(['cik', 'period'])``.
    * ``price_column`` contains numeric prices observed at each period end.
    * Period ordering within each ``cik`` is determined by the index sort order.

    Input schema
    ------------
    ``features_df``
        DataFrame indexed by ``(cik, period)`` containing at least ``price_column``.
    ``price_column``
        Column name used to compute ``P[t+1] / P[t] - 1``.

    Output schema
    -------------
    Series named ``next_period_return`` indexed by ``(cik, period)``. The final available
    period for each ``cik`` is excluded because no forward return exists.

    Raises
    ------
    KeyError
        If ``price_column`` is missing.
    ValueError
        If the index schema is invalid or if no next-period observations exist.
    """
    validate_multiindex_pandas_object(features_df, "features_df")
    validate_required_columns(features_df, [price_column], "features_df")

    price_series = features_df[price_column].copy()
    ordered_prices = price_series.sort_index(level=["cik", "period"])
    next_price = ordered_prices.groupby(level="cik", sort=False).shift(-1)

    if next_price.notna().sum() == 0:
        raise ValueError(
            "Unable to compute forward returns because no rows have a next period within the same cik."
        )

    forward_return = ((next_price / ordered_prices) - 1.0).dropna()
    forward_return = forward_return.reindex(forward_return.index.intersection(price_series.index))
    forward_return.name = "next_period_return"

    return forward_return
