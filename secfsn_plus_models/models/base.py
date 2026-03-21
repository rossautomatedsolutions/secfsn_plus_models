"""Base model abstractions for interchangeable estimator implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..utils import validate_multiindex_pandas_object


class BaseModel(ABC):
    """Abstract model interface for SECFSN+ estimators.

    Public model implementations must accept pandas inputs and return pandas outputs while
    preserving the ``(cik, period)`` MultiIndex on predictions.
    """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return a stable identifier for the model implementation."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the estimator using ``(cik, period)`` indexed pandas objects."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return predictions aligned to the exact index of ``X``."""

    def _validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate pandas training inputs shared by concrete models."""
        validate_multiindex_pandas_object(X, "X")
        validate_multiindex_pandas_object(y, "y")
        if not X.index.equals(y.index):
            raise ValueError("X and y must share the exact same MultiIndex during model.fit().")
        if X.empty or y.empty:
            raise ValueError("X and y must be non-empty during model.fit().")

    def _validate_prediction_input(self, X: pd.DataFrame) -> None:
        """Validate pandas prediction inputs shared by concrete models."""
        validate_multiindex_pandas_object(X, "X")
        if X.empty:
            raise ValueError("X must be non-empty during model.predict().")
