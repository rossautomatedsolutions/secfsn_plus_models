"""Automation helpers for CLI pipeline, diagnostics, and testing scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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
    """Create a deterministic synthetic panel with weak-but-learnable predictive structure."""
    ciks = [f"{idx:04d}" for idx in range(1, 7)]
    periods = ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2", "2021Q3"]
    index = pd.MultiIndex.from_product([ciks, periods], names=["cik", "period"])
    rng = np.random.default_rng(20240323)
    regime_strength = np.array([0.40, 0.30, 0.0, 0.20, 0.18, 0.14], dtype=float)
    period_linear_coefficients = rng.uniform(-0.0030, 0.0030, size=(len(periods) - 1, 5))
    base_linear_coefficients = np.array([0.0125, -0.0080, 0.0090, 0.0065, -0.0055], dtype=float)
    predictive_feature_masks = np.zeros((len(periods) - 1, 5), dtype=float)
    signal_free_periods = {2}

    for period_idx in range(len(periods) - 1):
        predictive_count = int(rng.integers(3, 5))
        predictive_columns = rng.choice(5, size=predictive_count, replace=False)
        predictive_feature_masks[period_idx, predictive_columns] = 1.0

    rows: list[dict[str, float | int]] = []
    row_positions: dict[tuple[str, str], int] = {}
    signal_by_period: dict[int, list[float]] = {period_idx: [] for period_idx in range(len(periods) - 1)}
    returns_by_cik: dict[str, list[float]] = {cik: [] for cik in ciks}

    for cik_idx, cik in enumerate(ciks, start=1):
        sector_code = cik_idx % 3
        sector_tilt = (sector_code - 1) * 0.12
        value_anchor = rng.normal(loc=-0.10 + (0.08 * cik_idx), scale=0.28)
        quality_anchor = rng.normal(loc=0.06 * (cik_idx - 3), scale=0.22)
        momentum_state = rng.normal(loc=0.0, scale=0.35)
        stability_anchor = rng.normal(loc=0.10 * (cik_idx - 3), scale=0.18)

        for period_idx, period in enumerate(periods):
            time_drift = (period_idx - ((len(periods) - 1) / 2.0)) / len(periods)
            cyclical_component = float(np.sin((period_idx + 1) * 0.85 + (0.30 * cik_idx)))

            value_anchor = (0.88 * value_anchor) + rng.normal(scale=0.18)
            quality_anchor = (0.82 * quality_anchor) + rng.normal(scale=0.16)
            momentum_state = (0.58 * momentum_state) + rng.normal(scale=0.34)
            stability_anchor = (0.90 * stability_anchor) + rng.normal(scale=0.08)

            feature_growth = (
                0.25 * momentum_state
                + 0.16 * quality_anchor
                + 0.12 * cyclical_component
                + 0.08 * time_drift
                + rng.normal(scale=0.34)
            )
            feature_value = value_anchor + (0.18 * sector_tilt) + (0.08 * time_drift) + rng.normal(scale=0.42)
            feature_quality = (
                0.44 * quality_anchor
                + 0.18 * value_anchor
                + 0.12 * cyclical_component
                + rng.normal(scale=0.30)
            )
            feature_momentum = (
                0.52 * momentum_state
                + 0.10 * time_drift
                + 0.10 * sector_tilt
                + rng.normal(scale=0.38)
            )
            feature_stability = (
                stability_anchor
                - (0.12 * abs(momentum_state))
                + 0.12 * quality_anchor
                + rng.normal(scale=0.20)
            )

            noisy_mix = rng.normal(scale=0.18)
            rows.append(
                {
                    price_column: np.nan,
                    "feature_growth": float(feature_growth + noisy_mix),
                    "feature_value": float((0.82 * feature_value) + (0.18 * rng.normal())),
                    "feature_quality": float((0.88 * feature_quality) + (0.12 * rng.normal())),
                    "feature_momentum": float((0.78 * feature_momentum) + (0.22 * rng.normal())),
                    "feature_stability": float((0.86 * feature_stability) + (0.14 * rng.normal())),
                    "sector_code": sector_code,
                }
            )
            row_positions[(cik, period)] = len(rows) - 1

            if period_idx == len(periods) - 1:
                continue

            feature_vector = np.array(
                [
                    rows[-1]["feature_growth"],
                    rows[-1]["feature_value"],
                    rows[-1]["feature_quality"],
                    rows[-1]["feature_momentum"],
                    rows[-1]["feature_stability"],
                ],
                dtype=float,
            )

            coefficients = (base_linear_coefficients + period_linear_coefficients[period_idx]) * predictive_feature_masks[
                period_idx
            ]
            linear_signal = float(np.dot(coefficients, feature_vector))
            interaction_signal = 0.0035 * feature_vector[0] * feature_vector[2] - 0.0025 * feature_vector[1] * feature_vector[3]
            nonlinear_signal = 0.0035 * np.tanh(feature_vector[3]) + 0.0020 * np.sign(feature_vector[1]) * (
                feature_vector[4] ** 2
            )
            latent_signal = regime_strength[period_idx] * (linear_signal + interaction_signal + nonlinear_signal)
            if period_idx in signal_free_periods:
                latent_signal = 0.0
            signal_by_period[period_idx].append(latent_signal)

    if signal_by_period:
        noisy_returns_by_period: dict[int, np.ndarray] = {}
        for period_idx, period_signal_list in signal_by_period.items():
            period_signal = np.array(period_signal_list, dtype=float)
            signal_std = float(period_signal.std(ddof=0))
            noise_floor = 0.018 if period_idx in signal_free_periods else 0.012
            noise_scale = max(signal_std * float(rng.uniform(1.05, 1.45)), noise_floor)
            noise = rng.normal(loc=0.0, scale=noise_scale, size=len(ciks))
            period_returns = period_signal + noise

            if len(ciks) > 1 and float(rng.random()) < 0.08:
                shuffled_positions = rng.choice(len(ciks), size=2, replace=False)
                shuffled_returns = period_returns.copy()
                shuffled_returns[shuffled_positions] = period_returns[np.roll(shuffled_positions, 1)]
                period_returns = shuffled_returns

            noisy_returns_by_period[period_idx] = np.clip(period_returns, -0.18, 0.18)

        for cik_idx, cik in enumerate(ciks):
            returns_by_cik[cik] = [
                float(noisy_returns_by_period[period_idx][cik_idx]) for period_idx in range(len(periods) - 1)
            ]

    for cik_idx, cik in enumerate(ciks, start=1):
        price = 82.0 + (13.0 * cik_idx) + float(rng.normal(scale=4.0))
        for period_idx, period in enumerate(periods):
            rows[row_positions[(cik, period)]][price_column] = float(max(price, 5.0))
            if period_idx < len(periods) - 1:
                realized_return = returns_by_cik[cik][period_idx]
                price *= 1.0 + realized_return

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
