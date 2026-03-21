"""Model training and prediction orchestration."""

from __future__ import annotations

import pandas as pd

from .models.base import BaseModel
from .utils import copy_aligned_frame_and_series, validate_multiindex_pandas_object


def train_and_predict(
    model: BaseModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.Series:
    """Fit a model on training data and generate aligned predictions for the test set.

    Assumptions
    -----------
    * ``X_train``, ``y_train``, and ``X_test`` use ``MultiIndex(['cik', 'period'])``.
    * ``model`` implements the :class:`secfsn_plus_models.models.base.BaseModel` contract.

    Input schema
    ------------
    ``X_train``
        Training feature dataframe indexed by ``(cik, period)``.
    ``y_train``
        Training label series indexed by ``(cik, period)``.
    ``X_test``
        Test feature dataframe indexed by ``(cik, period)``.

    Output schema
    -------------
    Series named ``prediction`` indexed exactly like ``X_test``.

    Raises
    ------
    ValueError
        If training or test data are empty, or if model predictions fail to preserve index alignment.
    """
    aligned_X_train, aligned_y_train = copy_aligned_frame_and_series(
        X_train,
        y_train,
        object_names=("X_train", "y_train"),
    )
    validate_multiindex_pandas_object(X_test, "X_test")

    if aligned_X_train.empty or aligned_y_train.empty:
        raise ValueError("Training data must contain at least one aligned row.")
    if X_test.empty:
        raise ValueError("X_test must contain at least one row.")

    model.fit(aligned_X_train, aligned_y_train)
    predictions = model.predict(X_test.copy())

    if not isinstance(predictions, pd.Series):
        raise TypeError("model.predict must return a pandas Series.")
    if not predictions.index.equals(X_test.index):
        raise ValueError("model.predict must preserve the exact X_test MultiIndex.")

    return predictions.rename("prediction")
