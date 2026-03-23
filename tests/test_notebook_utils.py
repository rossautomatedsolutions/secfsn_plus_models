from __future__ import annotations

from secfsn_plus_models.dataset import build_model_dataset
from secfsn_plus_models.labels import compute_next_period_return
from secfsn_plus_models.notebook_utils import (
    build_prediction_evaluation_frame,
    compare_model_runs,
    load_notebook_demo_artifacts,
    preview_aligned_dataset,
    summarize_model_comparison_metrics,
    summarize_dataset,
    summarize_period_coverage,
    summarize_period_metrics,
    summarize_split_artifacts,
)


def test_load_notebook_demo_artifacts_reuses_pipeline() -> None:
    demo = load_notebook_demo_artifacts(model_type="linear_regression", train_end_period="2020Q4")

    assert demo.pipeline.predictions.index.equals(demo.pipeline.X_test.index)
    assert demo.evaluation_summary.index.tolist() == ["IC", "RankIC", "DecileSpread"]
    assert "dataset" in demo.dataset_source


def test_dataset_summary_and_preview_include_labels(synthetic_panel) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")

    summary = summarize_dataset(synthetic_panel, labels)
    coverage = summarize_period_coverage(synthetic_panel, labels)
    preview = preview_aligned_dataset(synthetic_panel, labels, limit=4)

    assert summary.loc["dataset", "rows"] == len(synthetic_panel)
    assert summary.loc["dataset", "label_rows"] == len(labels)
    assert coverage.loc["2020Q1", "feature_rows"] == 4
    assert coverage.loc["2021Q3", "label_rows"] == 0
    assert preview.shape[0] == 4
    assert labels.name in preview.columns


def test_compare_model_runs_and_split_summary_return_expected_columns(synthetic_panel) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")
    X, y, _ = build_model_dataset(synthetic_panel, labels, exclude_columns=["price", "sector_code"])
    demo = load_notebook_demo_artifacts(model_type="linear_regression", train_end_period="2020Q4")

    comparison = compare_model_runs(X, y, train_end_period="2020Q4")
    split_summary = summarize_split_artifacts(demo.pipeline)

    assert set(comparison["model_name"]) == {"linear_regression", "random_forest"}
    assert set(
        [
            "IC",
            "RankIC",
            "DecileSpread",
            "IC_mean",
            "RankIC_mean",
            "prediction_mean",
            "prediction_std",
            "top_decile_return",
            "bottom_decile_return",
            "spread",
        ]
    ).issubset(comparison.columns)
    assert split_summary.loc["split", "train_rows"] == len(demo.pipeline.X_train)
    assert split_summary.loc["split", "test_rows"] == len(demo.pipeline.X_test)


def test_summarize_model_comparison_metrics_returns_requested_fields() -> None:
    demo = load_notebook_demo_artifacts(model_type="linear_regression", train_end_period="2020Q4")

    summary = summarize_model_comparison_metrics(
        "linear_regression",
        demo.pipeline.y_test,
        demo.pipeline.predictions,
        ic_by_period=demo.pipeline.ic_by_period,
        rank_ic_by_period=demo.pipeline.rank_ic_by_period,
    )

    assert summary["model_name"] == "linear_regression"
    assert summary.index.tolist() == [
        "model_name",
        "IC_mean",
        "RankIC_mean",
        "prediction_mean",
        "prediction_std",
        "top_decile_return",
        "bottom_decile_return",
        "spread",
    ]


def test_prediction_evaluation_helpers_add_rank_columns(synthetic_panel) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")
    X, y, _ = build_model_dataset(synthetic_panel, labels, exclude_columns=["price", "sector_code"])
    comparison = compare_model_runs(X, y, train_end_period="2020Q4", model_names=["linear_regression"])
    assert comparison.loc[0, "model_name"] == "linear_regression"

    demo = load_notebook_demo_artifacts(model_type="linear_regression", train_end_period="2020Q4")
    period_metrics = summarize_period_metrics(demo.pipeline.y_test, demo.pipeline.predictions)
    evaluation_frame = build_prediction_evaluation_frame(
        demo.pipeline.y_test,
        demo.pipeline.predictions,
        meta=demo.pipeline.meta,
    )

    assert set(["IC", "RankIC", "DecileSpread"]).issubset(period_metrics.columns)
    assert set(["actual", "prediction", "prediction_rank_desc", "actual_rank_desc", "prediction_bucket"]).issubset(
        evaluation_frame.columns
    )
    assert evaluation_frame.index.equals(demo.pipeline.predictions.index)
    assert "sector_code" in evaluation_frame.columns
