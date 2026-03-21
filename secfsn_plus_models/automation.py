"""Automation helpers for CLI pipeline, diagnostics, and testing scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import build_model_dataset
from .labels import compute_next_period_return
from .metrics import information_coefficient, rank_information_coefficient
from .models.registry import ModelRegistry
from .splits import time_based_split
from .trainer import train_and_predict
from .utils import validate_multiindex_pandas_object, validate_required_columns


@dataclass(frozen=True)
class PipelineArtifacts:
    """Materialized intermediate objects for the end-to-end modeling loop."""

    features_df: pd.DataFrame
    labels: pd.Series
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    predictions: pd.Series
    ic_by_period: pd.Series
    rank_ic_by_period: pd.Series


@dataclass(frozen=True)
class DiagnosticsReport:
    """Structured diagnostics results for a feature matrix and derived artifacts."""

    row_count: int
    column_count: int
    numeric_coverage: pd.Series
    label_count: int
    leakage_check_passed: bool
    alignment_check_passed: bool
    split_overlap_count: int
    train_rows: int
    test_rows: int


@dataclass(frozen=True)
class ResolvedConfig:
    """Validated runtime configuration used by CLI entrypoints."""

    price_column: str
    train_end_period: str | None
    model_type: str
    test_size: int | None
    feature_matrix_path: str | None


SUPPORTED_FILE_SUFFIXES = {".csv", ".parquet", ".pkl", ".pickle"}


def structured_step(title: str) -> None:
    """Print a consistently formatted step header."""
    print(f"=== {title} ===")


def resolve_runtime_config(config_module: Any) -> ResolvedConfig:
    """Read and validate supported configuration attributes from ``config.py``."""
    price_column = getattr(config_module, "price_column", None)
    train_end_period = getattr(config_module, "train_end_period", None)
    model_type = getattr(config_module, "model_type", "linear")
    test_size = getattr(config_module, "test_size", None)
    feature_matrix_path = getattr(config_module, "feature_matrix_path", None)

    if not isinstance(price_column, str) or not price_column.strip():
        raise ValueError("config.price_column must be a non-empty string.")

    normalized_model_type = {
        "linear": "linear_regression",
        "linear_regression": "linear_regression",
        "random_forest": "random_forest",
    }.get(str(model_type).strip().lower())
    if normalized_model_type is None:
        raise ValueError("config.model_type must be one of: linear, linear_regression, random_forest.")

    if train_end_period is not None and (not isinstance(train_end_period, str) or not train_end_period.strip()):
        raise ValueError("config.train_end_period must be None or a canonical quarterly string like '2020Q4'.")

    if test_size is not None:
        if not isinstance(test_size, int) or test_size <= 0:
            raise ValueError("config.test_size must be None or a positive integer.")

    if train_end_period is None and test_size is None:
        raise ValueError("Either config.train_end_period or config.test_size must be provided.")

    if feature_matrix_path is not None and not isinstance(feature_matrix_path, str):
        raise ValueError("config.feature_matrix_path must be None or a string path.")

    return ResolvedConfig(
        price_column=price_column.strip(),
        train_end_period=train_end_period.strip() if isinstance(train_end_period, str) else None,
        model_type=normalized_model_type,
        test_size=test_size,
        feature_matrix_path=feature_matrix_path,
    )


def generate_synthetic_feature_matrix(price_column: str) -> pd.DataFrame:
    """Create a deterministic MultiIndex feature matrix that always supports the CLI flows."""
    ciks = [f"{idx:04d}" for idx in range(1, 7)]
    periods = ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2", "2021Q3"]
    index = pd.MultiIndex.from_product([ciks, periods], names=["cik", "period"])

    rows: list[dict[str, float | int]] = []
    for cik_idx, _cik in enumerate(ciks, start=1):
        base_price = 75.0 + (18.0 * cik_idx)
        for period_idx, _period in enumerate(periods, start=1):
            rows.append(
                {
                    price_column: base_price + (4.5 * period_idx) + (1.25 * cik_idx * period_idx),
                    "feature_growth": float(period_idx),
                    "feature_value": float(cik_idx),
                    "feature_quality": float((cik_idx * period_idx) / 3.0),
                    "feature_momentum": float((period_idx**2) - cik_idx),
                    "feature_stability": float(cik_idx + (period_idx / 10.0)),
                    "sector_code": cik_idx % 3,
                }
            )

    return pd.DataFrame(rows, index=index).sort_index()


def _read_feature_matrix(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FILE_SUFFIXES:
        raise ValueError(
            f"Unsupported feature matrix format '{path.suffix}'. Supported formats: {sorted(SUPPORTED_FILE_SUFFIXES)}."
        )
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_pickle(path)

    if {"cik", "period"}.issubset(frame.columns):
        frame = frame.set_index(["cik", "period"])
    if not isinstance(frame.index, pd.MultiIndex):
        raise ValueError(
            "Loaded feature matrix must either already use MultiIndex(['cik', 'period']) or contain cik/period columns."
        )
    frame.index = frame.index.set_names(["cik", "period"])
    return frame.sort_index()


def load_feature_matrix(price_column: str, feature_matrix_path: str | None = None) -> tuple[pd.DataFrame, str]:
    """Load a real feature matrix when available, otherwise fall back to a synthetic dataset."""
    candidate_paths: list[Path] = []
    if feature_matrix_path:
        candidate_paths.append(Path(feature_matrix_path).expanduser())

    for candidate in candidate_paths:
        if candidate.exists():
            features_df = _read_feature_matrix(candidate)
            return features_df, f"loaded real dataset from {candidate}"

    fallback = generate_synthetic_feature_matrix(price_column=price_column)
    return fallback, "generated synthetic fallback dataset"


def validate_feature_matrix_assumptions(features_df: pd.DataFrame, price_column: str) -> pd.Series:
    """Validate structural assumptions for the input feature matrix."""
    validate_multiindex_pandas_object(features_df, "features_df")
    validate_required_columns(features_df, [price_column], "features_df")

    if features_df.empty:
        raise ValueError("Feature matrix is empty.")
    if features_df.index.has_duplicates:
        duplicate_rows = features_df.index[features_df.index.duplicated()].unique().tolist()
        raise ValueError(f"Feature matrix contains duplicate index rows: {duplicate_rows[:5]}.")

    numeric_columns = features_df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        raise ValueError("Feature matrix must contain at least one numeric feature column.")

    numeric_coverage = features_df.loc[:, numeric_columns].notna().mean().sort_index()
    if (numeric_coverage <= 0).any():
        bad_columns = numeric_coverage[numeric_coverage <= 0].index.tolist()
        raise ValueError(f"Numeric feature columns have zero non-null coverage: {bad_columns}.")

    return numeric_coverage


def validate_label_generation(features_df: pd.DataFrame, price_column: str, labels: pd.Series | None = None) -> pd.Series:
    """Validate alignment and next-period label construction with a no-leakage check."""
    labels = compute_next_period_return(features_df, price_column) if labels is None else labels.copy()
    validate_multiindex_pandas_object(labels, "labels")

    ordered_prices = features_df[price_column].sort_index(level=["cik", "period"])
    expected = ((ordered_prices.groupby(level="cik", sort=False).shift(-1) / ordered_prices) - 1.0).dropna()
    expected.name = "next_period_return"

    labels_aligned = labels.sort_index()
    expected_aligned = expected.sort_index()
    if not labels_aligned.index.equals(expected_aligned.index):
        raise ValueError("Generated labels are not correctly aligned to the expected next-period rows.")

    pd.testing.assert_series_equal(labels_aligned, expected_aligned, check_names=True, check_exact=False, rtol=1e-12)
    return labels_aligned


def resolve_train_end_period(
    features_df: pd.DataFrame,
    labels: pd.Series,
    train_end_period: str | None,
    test_size: int | None,
) -> str:
    """Choose the effective training cutoff from explicit config or from ``test_size``."""
    aligned_periods = pd.Index(features_df.index.intersection(labels.index).get_level_values("period")).unique().sort_values()
    if aligned_periods.empty:
        raise ValueError("Unable to resolve a training split because the aligned dataset is empty.")

    if train_end_period is not None:
        if train_end_period not in set(aligned_periods):
            raise ValueError(
                f"config.train_end_period={train_end_period!r} is not present in the aligned dataset periods {list(aligned_periods)}."
            )
        return train_end_period

    assert test_size is not None
    if test_size >= len(aligned_periods):
        raise ValueError(
            f"config.test_size={test_size} leaves no training periods; aligned periods are {list(aligned_periods)}."
        )
    return str(aligned_periods[-test_size])


def build_pipeline_artifacts(config: ResolvedConfig) -> PipelineArtifacts:
    """Execute the end-to-end pipeline and return all major intermediate artifacts."""
    features_df, _ = load_feature_matrix(config.price_column, config.feature_matrix_path)
    validate_feature_matrix_assumptions(features_df, config.price_column)

    labels = validate_label_generation(features_df, config.price_column)
    X, y, meta = build_model_dataset(features_df, labels, exclude_columns=[config.price_column, "sector_code"])

    if X.empty or y.empty:
        raise ValueError("Aligned modeling dataset is empty after feature/label assembly.")

    effective_train_end = resolve_train_end_period(features_df, labels, config.train_end_period, config.test_size)
    X_train, y_train, _, _, X_test, y_test = time_based_split(X, y, train_end_period=effective_train_end)

    if set(X_train.index).intersection(set(X_test.index)):
        raise ValueError("Invalid split: train and test datasets overlap.")

    model = ModelRegistry.create(config.model_type)
    predictions = train_and_predict(model, X_train, y_train, X_test)
    ic_by_period = information_coefficient(y_test, predictions)
    rank_ic_by_period = rank_information_coefficient(y_test, predictions)

    return PipelineArtifacts(
        features_df=features_df,
        labels=labels,
        X=X,
        y=y,
        meta=meta,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
        ic_by_period=ic_by_period,
        rank_ic_by_period=rank_ic_by_period,
    )


def run_diagnostics(config: ResolvedConfig) -> DiagnosticsReport:
    """Run validations required by the diagnostics CLI entrypoint."""
    features_df, _ = load_feature_matrix(config.price_column, config.feature_matrix_path)
    numeric_coverage = validate_feature_matrix_assumptions(features_df, config.price_column)
    labels = validate_label_generation(features_df, config.price_column)
    X, y, _ = build_model_dataset(features_df, labels, exclude_columns=[config.price_column, "sector_code"])

    if X.empty or y.empty:
        raise ValueError("Aligned modeling dataset is empty after dataset construction.")

    effective_train_end = resolve_train_end_period(features_df, labels, config.train_end_period, config.test_size)
    X_train, y_train, _, _, X_test, y_test = time_based_split(X, y, train_end_period=effective_train_end)
    overlap_count = len(set(X_train.index).intersection(set(X_test.index)))
    if overlap_count:
        raise ValueError("Invalid split: overlapping rows detected between train and test sets.")
    if not X_train.index.equals(y_train.index) or not X_test.index.equals(y_test.index):
        raise ValueError("Split alignment failure: X and y indices diverged within train or test sets.")

    return DiagnosticsReport(
        row_count=len(features_df),
        column_count=len(features_df.columns),
        numeric_coverage=numeric_coverage,
        label_count=len(labels),
        leakage_check_passed=True,
        alignment_check_passed=True,
        split_overlap_count=overlap_count,
        train_rows=len(X_train),
        test_rows=len(X_test),
    )
