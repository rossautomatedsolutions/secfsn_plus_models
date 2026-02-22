"""Model registry for model-name based instantiation."""

from __future__ import annotations

from typing import Any

from .base import BaseModel
from .linear import LinearRegressionModel
from .tree import RandomForestModel


class ModelRegistry:
    """Instantiate supported models by name."""

    _REGISTRY = {
        "linear_regression": LinearRegressionModel,
        "random_forest": RandomForestModel,
    }

    @classmethod
    def create(cls, model_name: str, **kwargs: Any) -> BaseModel:
        """Create model instance from a registered ``model_name``."""
        if model_name not in cls._REGISTRY:
            available = ", ".join(sorted(cls._REGISTRY))
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}.")
        return cls._REGISTRY[model_name](**kwargs)

    @classmethod
    def available_models(cls) -> list[str]:
        """Return sorted list of registered model names."""
        return sorted(cls._REGISTRY)
