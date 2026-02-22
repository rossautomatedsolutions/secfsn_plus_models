"""Layer 3 modeling package for SECFSN+ quarterly research workflows."""

from .dataset import build_model_dataset
from .evaluation import evaluate_cross_sectional_model
from .labels import compute_next_period_return
from .metrics import information_coefficient, rank_information_coefficient
from .splits import time_based_split
from .trainer import train_and_predict

__all__ = [
    "build_model_dataset",
    "compute_next_period_return",
    "time_based_split",
    "train_and_predict",
    "information_coefficient",
    "rank_information_coefficient",
    "evaluate_cross_sectional_model",
]
