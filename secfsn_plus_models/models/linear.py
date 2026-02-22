"""Linear-model implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.linear_model import LinearRegression

from .base import BaseModel


@dataclass
class LinearRegressionModel(BaseModel):
    """Wrapper around ``sklearn.linear_model.LinearRegression``."""

    fit_intercept: bool = True
    n_jobs: int | None = None
    _model: LinearRegression = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = LinearRegression(fit_intercept=self.fit_intercept, n_jobs=self.n_jobs)

    @property
    def model_id(self) -> str:
        return "linear_regression"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self._model.predict(X)
        return pd.Series(preds, index=X.index, name="prediction")
