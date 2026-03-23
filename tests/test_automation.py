from __future__ import annotations

import pandas as pd
import pytest

from secfsn_plus_models.automation import (
    build_pipeline_artifacts,
    generate_synthetic_feature_matrix,
    resolve_train_end_period,
    run_diagnostics,
    validate_feature_matrix_assumptions,
    validate_label_generation,
)
from secfsn_plus_models.labels import compute_next_period_return


class _Config:
    price_column = "price"
    train_end_period = "2020Q4"
    model_type = "linear_regression"
    test_size = None
    feature_matrix_path = None


def test_generate_synthetic_feature_matrix_returns_expected_index() -> None:
    features_df = generate_synthetic_feature_matrix(price_column="price")

    assert isinstance(features_df.index, pd.MultiIndex)
    assert list(features_df.index.names) == ["cik", "period"]
    assert "price" in features_df.columns
    assert len(features_df) > 0


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

    assert 0.03 <= ic_mean <= 0.08
    assert 0.05 <= rank_ic_mean <= 0.12
