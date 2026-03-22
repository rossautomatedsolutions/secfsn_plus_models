"""Layer 3 modeling package for SECFSN+ quarterly research workflows."""

from .dataset import build_model_dataset
from .evaluation import decile_spread, evaluate_cross_sectional_model
from .labels import compute_next_period_return
from .notebook_utils import (
    build_prediction_evaluation_frame,
    compare_model_runs,
    load_notebook_demo_artifacts,
    preview_aligned_dataset,
    summarize_dataset,
    summarize_period_coverage,
    summarize_period_metrics,
    summarize_split_artifacts,
)
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
    "decile_spread",
    "evaluate_cross_sectional_model",
    "load_notebook_demo_artifacts",
    "summarize_dataset",
    "summarize_period_coverage",
    "preview_aligned_dataset",
    "summarize_split_artifacts",
    "compare_model_runs",
    "summarize_period_metrics",
    "build_prediction_evaluation_frame",
]
