from __future__ import annotations

import pandas as pd
import pytest

from secfsn_plus_models.automation import (
    build_feature_stability_summary,
    build_robustness_summary,
    build_pipeline_artifacts,
    generate_synthetic_feature_matrix,
    interpret_feature_stability,
    interpret_robustness_summary,
    load_feature_matrix,
    resolve_runtime_config,
    resolve_train_end_period,
    run_robustness_analysis,
    run_diagnostics,
    summarize_run_metrics,
    validate_feature_matrix_assumptions,
    validate_label_generation,
)
from secfsn_plus_models.labels import compute_next_period_return


class _Config:
    price_column = "price"
    train_end_period = "2020Q4"
    model_type = "linear_regression"
    test_size = None
    USE_SYNTHETIC_DATA = True
    DATA_PATH = None
    feature_matrix_path = None
    synthetic_seed = 20240323
    robustness_runs = 4


def test_generate_synthetic_feature_matrix_returns_expected_index() -> None:
    features_df = generate_synthetic_feature_matrix(price_column="price")

    assert isinstance(features_df.index, pd.MultiIndex)
    assert list(features_df.index.names) == ["cik", "period"]
    assert "price" in features_df.columns
    assert len(features_df) > 0


def test_generate_synthetic_feature_matrix_is_deterministic_and_centered() -> None:
    first = generate_synthetic_feature_matrix(price_column="price", seed=20240323)
    second = generate_synthetic_feature_matrix(price_column="price", seed=20240323)
    labels = compute_next_period_return(first, "price")

    pd.testing.assert_frame_equal(first, second)
    assert first.shape == (42, 7)
    assert abs(float(labels.mean())) < 0.02
    assert float(labels.std()) > 0.015


def test_generate_synthetic_feature_matrix_changes_with_seed() -> None:
    baseline = generate_synthetic_feature_matrix(price_column="price", seed=20240323)
    shifted = generate_synthetic_feature_matrix(price_column="price", seed=20240324)

    assert baseline.shape == shifted.shape
    assert not baseline.equals(shifted)


def test_validate_feature_matrix_assumptions_returns_numeric_coverage(synthetic_panel: pd.DataFrame) -> None:
    coverage = validate_feature_matrix_assumptions(synthetic_panel, "price")

    assert "price" in coverage.index
    assert (coverage > 0).all()


def test_validate_feature_matrix_assumptions_rejects_duplicate_rows(synthetic_panel: pd.DataFrame) -> None:
    duplicated = pd.concat([synthetic_panel, synthetic_panel.iloc[[0]]])

    with pytest.raises(ValueError, match="duplicate index rows"):
        validate_feature_matrix_assumptions(duplicated, "price")


def test_validate_label_generation_matches_reference_formula(synthetic_panel: pd.DataFrame) -> None:
    labels = validate_label_generation(synthetic_panel, "price")
    expected = compute_next_period_return(synthetic_panel, "price")

    pd.testing.assert_series_equal(labels, expected)


def test_load_feature_matrix_accepts_in_memory_dataframe(synthetic_panel: pd.DataFrame) -> None:
    loaded, source = load_feature_matrix(price_column="price", features_df=synthetic_panel, use_synthetic_data=False)

    pd.testing.assert_frame_equal(loaded, synthetic_panel.sort_index())
    assert source == "loaded in-memory DataFrame input"


def test_load_feature_matrix_requires_real_dataset_when_synthetic_disabled(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Real dataset is required"):
        load_feature_matrix(
            price_column="price",
            feature_matrix_path=str(tmp_path / "missing.parquet"),
            use_synthetic_data=False,
        )


def test_resolve_runtime_config_rejects_missing_data_path_when_synthetic_disabled() -> None:
    class _MissingPathConfig:
        price_column = "price"
        train_end_period = "2020Q4"
        model_type = "linear_regression"
        test_size = None
        USE_SYNTHETIC_DATA = False
        DATA_PATH = None
        feature_matrix_path = None
        synthetic_seed = 20240323
        robustness_runs = 2

    with pytest.raises(ValueError, match="config.DATA_PATH"):
        resolve_runtime_config(_MissingPathConfig())


def test_resolve_train_end_period_uses_test_size_when_needed(synthetic_panel: pd.DataFrame) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")

    train_end = resolve_train_end_period(synthetic_panel, labels, train_end_period=None, test_size=2)

    assert train_end == "2021Q1"


def test_run_diagnostics_returns_clean_report() -> None:
    report = run_diagnostics(_Config())

    assert report.leakage_check_passed is True
    assert report.alignment_check_passed is True
    assert report.split_overlap_count == 0
    assert report.train_rows > 0
    assert report.test_rows > 0


def test_synthetic_pipeline_ic_mean_stays_in_target_signal_range() -> None:
    artifacts = build_pipeline_artifacts(_Config())
    ic_mean = float(artifacts.ic_by_period.mean())
    rank_ic_mean = float(artifacts.rank_ic_by_period.mean())

    assert 0.03 <= ic_mean <= 0.10
    assert 0.05 <= rank_ic_mean <= 0.15


def test_build_pipeline_artifacts_exposes_feature_importance() -> None:
    artifacts = build_pipeline_artifacts(_Config())

    assert artifacts.feature_importance.index.tolist() == artifacts.X_train.columns.tolist()
    assert artifacts.feature_importance.name == "importance"
    assert artifacts.feature_importance.notna().all()


def test_summarize_run_metrics_returns_expected_fields() -> None:
    artifacts = build_pipeline_artifacts(_Config())

    summary = summarize_run_metrics(artifacts)

    assert summary.index.tolist() == ["IC_mean", "RankIC_mean", "spread"]
    assert pd.notna(summary["spread"])


def test_run_robustness_analysis_collects_runs_and_summary() -> None:
    analysis = run_robustness_analysis(_Config(), model_names=["linear_regression", "random_forest"], run_count=3)

    assert len(analysis.per_run_metrics) == 6
    assert len(analysis.per_run_feature_importance) == 30
    assert set(analysis.per_run_metrics["seed"]) == {20240323, 20240324, 20240325}
    assert set(analysis.summary["model_name"]) == {"linear_regression", "random_forest"}
    assert set(analysis.feature_stability["model_name"]) == {"linear_regression", "random_forest"}
    assert {
        "IC_mean",
        "IC_std",
        "IC_min",
        "IC_max",
        "RankIC_mean",
        "RankIC_std",
        "RankIC_min",
        "RankIC_max",
        "spread_mean",
        "spread_std",
        "spread_min",
        "spread_max",
    }.issubset(set(analysis.summary.columns))
    assert {
        "feature_name",
        "mean_importance",
        "std_importance",
        "mean_abs_importance",
        "rank_stability",
    }.issubset(set(analysis.feature_stability.columns))


def test_build_feature_stability_summary_and_flags_capture_expected_signals() -> None:
    summary = build_feature_stability_summary(
        pd.DataFrame(
            [
                {"run_id": 1, "seed": 1, "model_name": "linear_regression", "feature_name": "stable", "importance": 0.30, "abs_importance": 0.30},
                {"run_id": 1, "seed": 1, "model_name": "linear_regression", "feature_name": "unstable", "importance": 0.40, "abs_importance": 0.40},
                {"run_id": 2, "seed": 2, "model_name": "linear_regression", "feature_name": "stable", "importance": 0.31, "abs_importance": 0.31},
                {"run_id": 2, "seed": 2, "model_name": "linear_regression", "feature_name": "unstable", "importance": -0.05, "abs_importance": 0.05},
                {"run_id": 3, "seed": 3, "model_name": "linear_regression", "feature_name": "stable", "importance": 0.29, "abs_importance": 0.29},
                {"run_id": 3, "seed": 3, "model_name": "linear_regression", "feature_name": "unstable", "importance": 0.55, "abs_importance": 0.55},
            ]
        )
    )

    warnings = interpret_feature_stability(summary)

    assert {"stable", "unstable"} == set(summary["feature_name"])
    assert any(warning.warning_type == "consistent_signal" and warning.feature_name == "stable" for warning in warnings)
    assert any(warning.warning_type == "unstable_feature" and warning.feature_name == "unstable" for warning in warnings)


def test_interpret_robustness_summary_flags_large_ranges() -> None:
    summary = build_robustness_summary(
        pd.DataFrame(
            [
                {"run_id": 1, "seed": 1, "model_name": "linear_regression", "IC_mean": 0.02, "RankIC_mean": 0.03, "spread": 0.01},
                {"run_id": 2, "seed": 2, "model_name": "linear_regression", "IC_mean": 0.28, "RankIC_mean": 0.35, "spread": 0.18},
                {"run_id": 3, "seed": 3, "model_name": "linear_regression", "IC_mean": -0.01, "RankIC_mean": -0.02, "spread": -0.02},
            ]
        )
    )

    warnings = interpret_robustness_summary(summary)

    assert warnings
    assert any(warning.warning_type == "instability" for warning in warnings)
