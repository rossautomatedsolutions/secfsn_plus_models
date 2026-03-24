from __future__ import annotations

import pandas as pd

from run_research_diagnostics import (
    ROBUSTNESS_DISPLAY_COLUMNS,
    ROBUSTNESS_SEEDS,
    _build_robustness_flags,
    _run_seeded_robustness_analysis,
)
from secfsn_plus_models.automation import ResolvedConfig


RUNTIME_CONFIG = ResolvedConfig(
    price_column="price",
    train_end_period="2020Q4",
    model_type="linear_regression",
    test_size=None,
    use_synthetic_data=True,
    data_path=None,
    feature_matrix_path=None,
    synthetic_seed=20240323,
    robustness_runs=10,
)


def test_run_seeded_robustness_analysis_uses_expected_seed_grid() -> None:
    per_run_metrics, summary = _run_seeded_robustness_analysis(
        RUNTIME_CONFIG,
        ["linear_regression", "random_forest"],
    )

    assert set(per_run_metrics["seed"]) == set(ROBUSTNESS_SEEDS)
    assert set(per_run_metrics["model_name"]) == {"linear_regression", "random_forest"}
    assert ROBUSTNESS_DISPLAY_COLUMNS == [
        "model_name",
        "IC_mean_avg",
        "IC_std",
        "IC_min",
        "IC_max",
        "RankIC_avg",
        "spread_avg",
    ]
    assert {"spread_std", "spread_min", "spread_max"}.issubset(summary.columns)


def test_build_robustness_flags_reports_requested_instability_messages() -> None:
    summary = pd.DataFrame(
        [
            {
                "model_name": "linear_regression",
                "IC_mean_avg": 0.05,
                "IC_std": 0.08,
                "IC_min": -0.02,
                "IC_max": 0.16,
                "RankIC_avg": 0.07,
                "spread_avg": 0.01,
                "spread_std": 0.05,
            }
        ]
    )

    warnings = _build_robustness_flags(summary)

    assert "WARNING: linear_regression unstable signal (IC_std > IC_mean_avg)." in warnings
    assert "WARNING: linear_regression inconsistent direction (IC_min < 0)." in warnings
    assert "WARNING: linear_regression unstable ranking (large spread variance)." in warnings
