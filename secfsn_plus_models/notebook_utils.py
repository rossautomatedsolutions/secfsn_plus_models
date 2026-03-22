"""Notebook-friendly helpers that reuse the tested SECFSN+ modeling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .automation import PipelineArtifacts, ResolvedConfig, build_pipeline_artifacts, load_feature_matrix
from .evaluation import decile_spread, evaluate_cross_sectional_model
from .metrics import information_coefficient, rank_information_coefficient
from .models.registry import ModelRegistry
from .splits import time_based_split
from .trainer import train_and_predict
from .utils import copy_aligned_series_pair, validate_multiindex_pandas_object


@dataclass(frozen=True)
class NotebookDemoArtifacts:
    """Materialized objects used by notebook walkthroughs and demos."""

    config: ResolvedConfig
    dataset_source: str
    pipeline: PipelineArtifacts
    evaluation_summary: pd.Series

    @property
    def X(self) -> pd.DataFrame:
        """Expose the aligned model feature matrix for notebook convenience."""
        return self.pipeline.X

    @property
    def y(self) -> pd.Series:
        """Expose the aligned model label series for notebook convenience."""
        return self.pipeline.y


DEFAULT_NOTEBOOK_MODELS = ["linear_regression", "random_forest"]


def load_notebook_demo_artifacts(
    *,
    price_column: str = "price",
    train_end_period: str | None = "2020Q4",
    model_type: str = "linear_regression",
    test_size: int | None = None,
    feature_matrix_path: str | None = None,
) -> NotebookDemoArtifacts:
    """Build notebook-ready artifacts using the repository's production pipeline.

    This helper intentionally delegates to :func:`build_pipeline_artifacts` so notebooks do not
    duplicate feature loading, label generation, dataset assembly, splitting, training, or metric
    logic.
    """
    config = ResolvedConfig(
        price_column=price_column,
        train_end_period=train_end_period,
        model_type=model_type,
        test_size=test_size,
        feature_matrix_path=feature_matrix_path,
    )
    _, dataset_source = load_feature_matrix(price_column, feature_matrix_path)
    pipeline = build_pipeline_artifacts(config)
    evaluation_summary = evaluate_cross_sectional_model(pipeline.y_test, pipeline.predictions)
    return NotebookDemoArtifacts(
        config=config,
        dataset_source=dataset_source,
        pipeline=pipeline,
        evaluation_summary=evaluation_summary,
    )


def summarize_dataset(features_df: pd.DataFrame, labels: pd.Series | None = None) -> pd.DataFrame:
    """Return a one-row dataset summary table for notebook display."""
    validate_multiindex_pandas_object(features_df, "features_df")

    summary: dict[str, Any] = {
        "rows": len(features_df),
        "columns": len(features_df.columns),
        "cik_count": int(features_df.index.get_level_values("cik").nunique()),
        "period_count": int(features_df.index.get_level_values("period").nunique()),
        "numeric_columns": int(len(features_df.select_dtypes(include=["number"]).columns)),
    }

    if labels is not None:
        validate_multiindex_pandas_object(labels, "labels")
        aligned_features = features_df.index.intersection(labels.index)
        summary["label_rows"] = len(labels)
        summary["aligned_rows"] = len(aligned_features)
        summary["label_name"] = labels.name

    return pd.DataFrame([summary], index=pd.Index(["dataset"], name="summary"))


def summarize_period_coverage(features_df: pd.DataFrame, labels: pd.Series | None = None) -> pd.DataFrame:
    """Summarize row coverage by period for features and optional labels."""
    validate_multiindex_pandas_object(features_df, "features_df")

    coverage = pd.DataFrame(
        {
            "feature_rows": features_df.groupby(level="period", sort=False).size(),
            "feature_ciks": features_df.groupby(level="period", sort=False)
            .apply(lambda frame: frame.index.get_level_values("cik").nunique())
            .astype(int),
        }
    )

    if labels is not None:
        validate_multiindex_pandas_object(labels, "labels")
        coverage["label_rows"] = labels.groupby(level="period", sort=False).size()

    return coverage.fillna(0).astype(int)


def preview_aligned_dataset(features_df: pd.DataFrame, labels: pd.Series, limit: int = 10) -> pd.DataFrame:
    """Join features and labels into a compact preview table for notebooks."""
    if limit <= 0:
        raise ValueError("limit must be a positive integer.")

    validate_multiindex_pandas_object(features_df, "features_df")
    validate_multiindex_pandas_object(labels, "labels")

    preview = features_df.join(labels.rename(labels.name or "label"), how="inner")
    preview = preview.sort_index(level=["period", "cik"])
    return preview.head(limit).copy()


def summarize_split_artifacts(artifacts: PipelineArtifacts) -> pd.DataFrame:
    """Return notebook-displayable train/test split statistics."""
    return pd.DataFrame(
        [
            {
                "train_rows": len(artifacts.X_train),
                "train_periods": int(artifacts.X_train.index.get_level_values("period").nunique()),
                "test_rows": len(artifacts.X_test),
                "test_periods": int(artifacts.X_test.index.get_level_values("period").nunique()),
                "feature_columns": len(artifacts.X.columns),
                "meta_columns": len(artifacts.meta.columns),
            }
        ],
        index=pd.Index(["split"], name="summary"),
    )


def compare_model_runs(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_end_period: str,
    model_names: list[str] | None = None,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Run multiple registered models on the same split and summarize results."""
    model_names = model_names or list(DEFAULT_NOTEBOOK_MODELS)
    model_params = model_params or {}

    X_train, y_train, _, _, X_test, y_test = time_based_split(X, y, train_end_period=train_end_period)

    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        model = ModelRegistry.create(model_name, **model_params.get(model_name, {}))
        predictions = train_and_predict(model, X_train, y_train, X_test)
        evaluation_summary = evaluate_cross_sectional_model(y_test, predictions)
        rows.append(
            {
                "model_name": model_name,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "IC": float(evaluation_summary["IC"]),
                "RankIC": float(evaluation_summary["RankIC"]),
                "DecileSpread": float(evaluation_summary["DecileSpread"]),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["RankIC", "IC"], ascending=False).reset_index(drop=True)


def summarize_period_metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """Combine period-level IC, RankIC, and decile spread into one table."""
    ic = information_coefficient(y_true, y_pred)
    rank_ic = rank_information_coefficient(y_true, y_pred)
    spread = decile_spread(y_true, y_pred)
    return pd.concat([ic.rename("IC"), rank_ic.rename("RankIC"), spread.rename("DecileSpread")], axis=1)


def build_prediction_evaluation_frame(
    y_true: pd.Series,
    y_pred: pd.Series,
    meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a notebook-friendly prediction review table with per-period ranks."""
    true_aligned, pred_aligned = copy_aligned_series_pair(
        y_true,
        y_pred,
        object_names=("y_true", "y_pred"),
        require_non_empty=True,
    )
    frame = pd.DataFrame({"actual": true_aligned, "prediction": pred_aligned})

    frame["prediction_rank_desc"] = frame.groupby(level="period", sort=False)["prediction"].rank(
        ascending=False,
        method="first",
    )
    frame["actual_rank_desc"] = frame.groupby(level="period", sort=False)["actual"].rank(
        ascending=False,
        method="first",
    )
    frame["prediction_percentile"] = frame.groupby(level="period", sort=False)["prediction"].rank(
        pct=True,
        method="average",
    )

    def _label_bucket(group: pd.Series) -> pd.Series:
        bucket_count = min(10, int(group.nunique()))
        if bucket_count < 2:
            return pd.Series(["single_bucket"] * len(group), index=group.index, dtype="object")

        bucket_ids = pd.qcut(group.rank(method="first"), q=bucket_count, labels=False, duplicates="drop")
        return bucket_ids.map(lambda value: f"Q{int(value) + 1}" if pd.notna(value) else "unassigned")

    frame["prediction_bucket"] = frame.groupby(level="period", sort=False)["prediction"].transform(_label_bucket)

    if meta is not None:
        validate_multiindex_pandas_object(meta, "meta")
        frame = frame.join(meta.reindex(frame.index), how="left")

    return frame.copy()
