"""Base model interface for pluggable model implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    """Abstract interface for all modeling-layer estimators."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Unique model identifier used in configuration and tracking."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Fit the model to training data."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions aligned to ``X`` index."""
