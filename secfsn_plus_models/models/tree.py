"""Tree-based model implementations."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel


@dataclass
class RandomForestModel(BaseModel):
    """Wrapper around ``sklearn.ensemble.RandomForestRegressor``."""

    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int | None = None
    n_jobs: int | None = None
    _model: RandomForestRegressor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    @property
    def model_id(self) -> str:
        return "random_forest"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        preds = self._model.predict(X)
        return pd.Series(preds, index=X.index, name="prediction")
