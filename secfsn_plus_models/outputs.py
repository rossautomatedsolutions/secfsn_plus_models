"""Canonical downstream output packaging helpers for Layer 04 model predictions."""

from __future__ import annotations

import pandas as pd

from .utils import validate_multiindex_pandas_object


def build_model_output_frame(
    predictions: pd.Series,
    *,
    higher_prediction_more_risky: bool,
    by_period: bool = True,
    prediction_col: str = "model_prediction",
    expected_return_col: str = "model_expected_return",
    risk_probability_col: str = "model_risk_probability",
) -> pd.DataFrame:
    """Package aligned model predictions into the canonical Layer 04 handoff shape.

    Parameters
    ----------
    predictions
        Prediction series indexed by ``MultiIndex(['cik', 'period'])``.
    higher_prediction_more_risky
        Direction flag used to map predictions into a rank-normalized risk score.
        For expected-return models, pass ``False`` so lower expected return maps to
        higher ``model_risk_probability``.
    by_period
        Whether to compute risk percentiles independently within each period.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed exactly like ``predictions`` with canonical model output
        columns suitable for downstream Layer 05 consumption.
    """
    validate_multiindex_pandas_object(predictions, "predictions")

    prediction_values = pd.to_numeric(predictions.copy(), errors="coerce").rename(prediction_col)
    output = pd.DataFrame(index=prediction_values.index)
    output[prediction_col] = prediction_values
    output[expected_return_col] = prediction_values

    if by_period:
        grouped = prediction_values.groupby(level="period", sort=False)
        risk_probability = grouped.rank(
            method="average",
            pct=True,
            ascending=higher_prediction_more_risky,
        )
    else:
        risk_probability = prediction_values.rank(
            method="average",
            pct=True,
            ascending=higher_prediction_more_risky,
        )

    output[risk_probability_col] = risk_probability.astype(float)
    return output
