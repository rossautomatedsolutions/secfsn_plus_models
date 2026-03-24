"""Notebook-friendly helpers that reuse the tested SECFSN+ modeling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .automation import PipelineArtifacts, ResolvedConfig, build_pipeline_artifacts, load_feature_matrix
from .evaluation import decile_return_summary, decile_spread, evaluate_cross_sectional_model
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
COMPARISON_METRIC_COLUMNS = [
    "IC_mean",
    "RankIC_mean",
    "prediction_mean",
    "prediction_std",
    "top_decile_return",
    "bottom_decile_return",
    "spread",
]


def summarize_model_comparison_metrics(
    model_name: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    ic_by_period: pd.Series | None = None,
    rank_ic_by_period: pd.Series | None = None,
) -> pd.Series:
    """Return compact comparison metrics for one model run."""
    ic_by_period = information_coefficient(y_true, y_pred) if ic_by_period is None else ic_by_period.copy()
    rank_ic_by_period = (
        rank_information_coefficient(y_true, y_pred) if rank_ic_by_period is None else rank_ic_by_period.copy()
    )
    decile_summary = decile_return_summary(y_true, y_pred)
    prediction_values = y_pred.dropna()

    summary = pd.Series(
        {
            "model_name": model_name,
            "IC_mean": float(ic_by_period.mean()) if not ic_by_period.empty else float("nan"),
            "RankIC_mean": float(rank_ic_by_period.mean()) if not rank_ic_by_period.empty else float("nan"),
            "prediction_mean": float(prediction_values.mean()) if not prediction_values.empty else float("nan"),
            "prediction_std": float(prediction_values.std(ddof=1)) if len(prediction_values) > 1 else float("nan"),
            "top_decile_return": float(decile_summary["top_decile_return"]),
            "bottom_decile_return": float(decile_summary["bottom_decile_return"]),
            "spread": float(decile_summary["spread"]),
        },
        dtype="object",
        name=model_name,
    )
    return summary


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
        use_synthetic_data=feature_matrix_path is None,
        data_path=feature_matrix_path,
        feature_matrix_path=feature_matrix_path,
        synthetic_seed=20240323,
        robustness_runs=10,
    )
    _, dataset_source = load_feature_matrix(
        price_column=price_column,
        feature_matrix_path=feature_matrix_path,
        use_synthetic_data=config.use_synthetic_data,
        synthetic_seed=config.synthetic_seed,
    )
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
        row = summarize_model_comparison_metrics(model_name, y_test, predictions).to_dict()
        row.update(
            {
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "IC": row["IC_mean"],
                "RankIC": row["RankIC_mean"],
                "DecileSpread": row["spread"],
            }
        )
        rows.append(row)

    comparison = pd.DataFrame(rows)
    return comparison.sort_values(
        by=["RankIC_mean", "IC_mean", "model_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


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
