"""Training orchestration utilities for model fitting and prediction."""

from __future__ import annotations

import pandas as pd

from .models.base import BaseModel


def train_and_predict(
    model: BaseModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.Series:
    """Fit a model and return index-aligned predictions for ``X_test``."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preds.reindex(X_test.index)
