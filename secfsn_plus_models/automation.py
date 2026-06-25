"""Automation helpers for CLI pipeline, diagnostics, and testing scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import build_model_dataset
from .evaluation import decile_return_summary
from .labels import compute_next_period_return
from .metrics import information_coefficient, rank_information_coefficient
from .models.registry import ModelRegistry
from .splits import time_based_split
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
    feature_importance: pd.Series


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
    use_synthetic_data: bool
    data_path: str | None
    feature_matrix_path: str | None
    synthetic_seed: int
    robustness_runs: int


@dataclass(frozen=True)
class RobustnessWarning:
    """Interpretation flag emitted when robustness metrics indicate instability."""

    model_name: str
    metric_name: str
    warning_type: str
    message: str


@dataclass(frozen=True)
class FeatureStabilityWarning:
    """Interpretation flag emitted when feature importance looks unstable or persistent."""

    model_name: str
    feature_name: str
    warning_type: str
    message: str


@dataclass(frozen=True)
class RobustnessAnalysis:
    """Per-run metrics, summary statistics, and warnings for robustness evaluation."""

    per_run_metrics: pd.DataFrame
    summary: pd.DataFrame
    warnings: list[RobustnessWarning]
    per_run_feature_importance: pd.DataFrame
    feature_stability: pd.DataFrame
    feature_warnings: list[FeatureStabilityWarning]


SUPPORTED_FILE_SUFFIXES = {".csv", ".parquet", ".pkl", ".pickle"}
DEFAULT_SYNTHETIC_SEED = 20240323
DEFAULT_ROBUSTNESS_RUNS = 10
HIGH_VARIANCE_RULES = {
    "IC_mean": {"std_floor": 0.035, "range_floor": 0.12},
    "RankIC_mean": {"std_floor": 0.050, "range_floor": 0.18},
    "spread": {"std_floor": 0.030, "range_floor": 0.10},
}
FEATURE_STABILITY_STD_RATIO = 0.75
FEATURE_SIGNAL_RANK_STD_MAX = 1.5


def structured_step(title: str) -> None:
    """Print a consistently formatted step header."""
    print(f"=== {title} ===")


def _config_use_synthetic_data(config: Any) -> bool:
    """Read synthetic-data preference from resolved or legacy config objects."""
    value = getattr(config, "use_synthetic_data", None)
    if value is None:
        value = getattr(config, "USE_SYNTHETIC_DATA", None)
    if value is None:
        return False
    return bool(value)


def _config_feature_matrix_path(config: Any) -> str | None:
    """Read the preferred feature-matrix path from resolved or legacy config objects."""
    value = getattr(config, "feature_matrix_path", None)
    if value is None:
        value = getattr(config, "data_path", None)
    if value is None:
        value = getattr(config, "DATA_PATH", None)
    return value


def resolve_runtime_config(config_module: Any) -> ResolvedConfig:
    """Read and validate supported configuration attributes from ``config.py``."""
    price_column = getattr(config_module, "price_column", None)
    train_end_period = getattr(config_module, "train_end_period", None)
    model_type = getattr(config_module, "model_type", "linear")
    test_size = getattr(config_module, "test_size", None)
    use_synthetic_data = getattr(config_module, "USE_SYNTHETIC_DATA", None)
    if use_synthetic_data is None:
        use_synthetic_data = getattr(config_module, "use_synthetic_data", None)
    if use_synthetic_data is None:
        use_synthetic_data = False

    data_path = getattr(config_module, "DATA_PATH", None)
    if data_path is None:
        data_path = getattr(config_module, "data_path", None)

    feature_matrix_path = getattr(config_module, "feature_matrix_path", None)
    if feature_matrix_path is None:
        feature_matrix_path = data_path

    synthetic_seed = getattr(config_module, "synthetic_seed", DEFAULT_SYNTHETIC_SEED)
    robustness_runs = getattr(config_module, "robustness_runs", DEFAULT_ROBUSTNESS_RUNS)

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

    if not isinstance(use_synthetic_data, bool):
        raise ValueError("config.USE_SYNTHETIC_DATA must be a boolean.")
    if data_path is not None and not isinstance(data_path, str):
        raise ValueError("config.DATA_PATH must be None or a string path.")
    if feature_matrix_path is not None and not isinstance(feature_matrix_path, str):
        raise ValueError("config.feature_matrix_path must be None or a string path.")
    if not use_synthetic_data and not feature_matrix_path:
        raise ValueError("config.DATA_PATH (or config.feature_matrix_path) is required when USE_SYNTHETIC_DATA=False.")
    if not isinstance(synthetic_seed, int):
        raise ValueError("config.synthetic_seed must be an integer.")
    if not isinstance(robustness_runs, int) or robustness_runs <= 0:
        raise ValueError("config.robustness_runs must be a positive integer.")

    return ResolvedConfig(
        price_column=price_column.strip(),
        train_end_period=train_end_period.strip() if isinstance(train_end_period, str) else None,
        model_type=normalized_model_type,
        test_size=test_size,
        use_synthetic_data=use_synthetic_data,
        data_path=feature_matrix_path,
        feature_matrix_path=feature_matrix_path,
        synthetic_seed=synthetic_seed,
        robustness_runs=robustness_runs,
    )


def generate_synthetic_feature_matrix(price_column: str, seed: int = DEFAULT_SYNTHETIC_SEED) -> pd.DataFrame:
    """Create a deterministic synthetic panel with weak-but-learnable predictive structure."""
    ciks = [f"{idx:04d}" for idx in range(1, 7)]
    periods = ["2020Q1", "2020Q2", "2020Q3", "2020Q4", "2021Q1", "2021Q2", "2021Q3"]
    index = pd.MultiIndex.from_product([ciks, periods], names=["cik", "period"])
    rng = np.random.default_rng(seed)
    feature_names = [
        "feature_growth",
        "feature_value",
        "feature_quality",
        "feature_momentum",
        "feature_stability",
    ]
    base_linear_coefficients = np.array([0.0055, -0.0070, 0.0095, 0.0040, 0.0065], dtype=float)
    regime_templates = [
        {
            "active_columns": np.array([1, 0, 1, 0, 1], dtype=float),
            "linear_shift": np.array([0.0080, 0.0000, 0.0065, 0.0000, 0.0030], dtype=float),
            "interaction_pair": (0, 2),
            "interaction_weight": 0.0048,
            "piecewise_column": 0,
            "piecewise_threshold": 0.18,
            "piecewise_weight": 0.0042,
            "quadratic_column": 4,
            "quadratic_weight": 0.0018,
            "signal_scale": 0.42,
            "noise_scale_floor": 0.018,
            "rank_mix": 0.20,
        },
        {
            "active_columns": np.array([0, 0, 0, 1, 1], dtype=float),
            "linear_shift": np.array([0.0000, 0.0000, 0.0000, 0.0105, 0.0040], dtype=float),
            "interaction_pair": (3, 4),
            "interaction_weight": 0.0038,
            "piecewise_column": 3,
            "piecewise_threshold": 0.10,
            "piecewise_weight": 0.0045,
            "quadratic_column": 1,
            "quadratic_weight": -0.0014,
            "signal_scale": 0.32,
            "noise_scale_floor": 0.020,
            "rank_mix": 0.22,
        },
        {
            "active_columns": np.array([0, 1, 0, 0, 1], dtype=float),
            "linear_shift": np.array([0.0000, -0.0035, 0.0000, 0.0000, 0.0025], dtype=float),
            "interaction_pair": (1, 4),
            "interaction_weight": 0.0016,
            "piecewise_column": 4,
            "piecewise_threshold": 0.05,
            "piecewise_weight": 0.0014,
            "quadratic_column": 2,
            "quadratic_weight": 0.0008,
            "signal_scale": 0.10,
            "noise_scale_floor": 0.024,
            "rank_mix": 0.32,
        },
        {
            "active_columns": np.array([1, 1, 0, 1, 0], dtype=float),
            "linear_shift": np.array([0.0045, -0.0060, 0.0000, 0.0050, 0.0000], dtype=float),
            "interaction_pair": (0, 3),
            "interaction_weight": 0.0030,
            "piecewise_column": 1,
            "piecewise_threshold": 0.08,
            "piecewise_weight": -0.0030,
            "quadratic_column": 0,
            "quadratic_weight": 0.0014,
            "signal_scale": 0.24,
            "noise_scale_floor": 0.019,
            "rank_mix": 0.24,
        },
        {
            "active_columns": np.array([0, 0, 1, 1, 1], dtype=float),
            "linear_shift": np.array([0.0000, 0.0000, 0.0055, 0.0050, 0.0040], dtype=float),
            "interaction_pair": (2, 3),
            "interaction_weight": 0.0032,
            "piecewise_column": 2,
            "piecewise_threshold": 0.16,
            "piecewise_weight": 0.0024,
            "quadratic_column": 3,
            "quadratic_weight": 0.0012,
            "signal_scale": 0.18,
            "noise_scale_floor": 0.018,
            "rank_mix": 0.18,
        },
        {
            "active_columns": np.array([0, 1, 1, 0, 1], dtype=float),
            "linear_shift": np.array([0.0000, -0.0030, 0.0050, 0.0000, 0.0055], dtype=float),
            "interaction_pair": (2, 4),
            "interaction_weight": 0.0028,
            "piecewise_column": 1,
            "piecewise_threshold": 0.04,
            "piecewise_weight": 0.0022,
            "quadratic_column": 4,
            "quadratic_weight": 0.0016,
            "signal_scale": 0.16,
            "noise_scale_floor": 0.018,
            "rank_mix": 0.16,
        },
    ]

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

            feature_vector = np.array([rows[-1][feature_name] for feature_name in feature_names], dtype=float)
            regime = regime_templates[period_idx]
            active_mask = regime["active_columns"].copy()
            if float(rng.random()) < 0.40:
                active_column = int(rng.integers(0, len(feature_names)))
                active_mask[active_column] = 0.0
            active_mask = np.where(active_mask.sum() == 0.0, regime["active_columns"], active_mask)

            backbone_signal = float(np.dot(base_linear_coefficients, feature_vector))
            regime_linear_signal = float(np.dot((regime["linear_shift"] * active_mask), feature_vector))

            left_idx, right_idx = regime["interaction_pair"]
            interaction_signal = float(
                regime["interaction_weight"] * feature_vector[left_idx] * feature_vector[right_idx] * active_mask[left_idx] * active_mask[right_idx]
            )

            piecewise_idx = regime["piecewise_column"]
            piecewise_input = feature_vector[piecewise_idx] - regime["piecewise_threshold"]
            piecewise_signal = float(regime["piecewise_weight"] * max(piecewise_input, 0.0) * active_mask[piecewise_idx])

            quadratic_idx = regime["quadratic_column"]
            quadratic_signal = float(
                regime["quadratic_weight"] * ((feature_vector[quadratic_idx] ** 2) - 0.5) * active_mask[quadratic_idx]
            )

            nonlinear_backbone = 0.0022 * np.tanh(feature_vector[2]) - 0.0018 * np.tanh(feature_vector[1] * feature_vector[3])
            latent_signal = backbone_signal + (regime["signal_scale"] * (regime_linear_signal + interaction_signal + piecewise_signal + quadratic_signal)) + nonlinear_backbone
            signal_by_period[period_idx].append(latent_signal)

    if signal_by_period:
        noisy_returns_by_period: dict[int, np.ndarray] = {}
        for period_idx, period_signal_list in signal_by_period.items():
            period_signal = np.array(period_signal_list, dtype=float)
            regime = regime_templates[period_idx]
            centered_signal = period_signal - float(period_signal.mean())
            signal_std = float(centered_signal.std(ddof=0))
            noise_scale = max(signal_std * float(rng.uniform(1.25, 1.55)), regime["noise_scale_floor"])
            heteroskedastic_scale = rng.uniform(0.85, 1.20, size=len(ciks))
            noise = rng.normal(loc=0.0, scale=noise_scale * heteroskedastic_scale, size=len(ciks))
            period_returns = centered_signal + noise

            if len(ciks) > 2:
                ranks = np.argsort(np.argsort(period_returns))
                rank_noise = rng.normal(loc=0.0, scale=regime["rank_mix"], size=len(ciks))
                perturbed_rank = np.clip(ranks.astype(float) + rank_noise, 0.0, len(ciks) - 1.0)
                order = np.argsort(perturbed_rank)
                sorted_returns = np.sort(period_returns)
                period_returns = np.empty_like(period_returns)
                period_returns[order] = sorted_returns

            if len(ciks) > 1 and float(rng.random()) < 0.55:
                swap_positions = rng.choice(len(ciks), size=2, replace=False)
                swap_weight = float(rng.uniform(0.25, 0.40))
                mixed_returns = period_returns.copy()
                a_idx, b_idx = int(swap_positions[0]), int(swap_positions[1])
                mixed_returns[a_idx] = ((1.0 - swap_weight) * period_returns[a_idx]) + (swap_weight * period_returns[b_idx])
                mixed_returns[b_idx] = ((1.0 - swap_weight) * period_returns[b_idx]) + (swap_weight * period_returns[a_idx])
                period_returns = mixed_returns

            noisy_returns_by_period[period_idx] = np.clip(period_returns * 1.30, -0.20, 0.20)

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


def load_feature_matrix(
    price_column: str,
    features_df: pd.DataFrame | None = None,
    feature_matrix_path: str | None = None,
    use_synthetic_data: bool = True,
    synthetic_seed: int = DEFAULT_SYNTHETIC_SEED,
) -> tuple[pd.DataFrame, str]:
    """Load a feature matrix from DataFrame/path, or synthesize one when enabled."""
    if features_df is not None:
        validate_multiindex_pandas_object(features_df, "features_df")
        normalized = features_df.copy()
        normalized.index = normalized.index.set_names(["cik", "period"])
        return normalized.sort_index(), "loaded in-memory DataFrame input"

    candidate_paths: list[Path] = []
    if feature_matrix_path:
        candidate_paths.append(Path(feature_matrix_path).expanduser())

    for candidate in candidate_paths:
        if candidate.exists():
            features_df = _read_feature_matrix(candidate)
            return features_df, f"loaded real dataset from {candidate}"

    if not use_synthetic_data:
        required_path = feature_matrix_path if feature_matrix_path else "<unspecified>"
        raise FileNotFoundError(f"Real dataset is required but was not found at {required_path}.")

    fallback = generate_synthetic_feature_matrix(price_column=price_column, seed=synthetic_seed)
    return fallback, f"generated synthetic fallback dataset (seed={synthetic_seed})"


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
    if labels_aligned.empty:
        raise ValueError("Insufficient history to compute forward returns: every cik needs at least two periods.")
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


def build_pipeline_artifacts(
    config: ResolvedConfig,
    *,
    features_df: pd.DataFrame | None = None,
    synthetic_seed: int | None = None,
    model_type_override: str | None = None,
) -> PipelineArtifacts:
    """Execute the end-to-end pipeline and return all major intermediate artifacts."""
    effective_seed = config.synthetic_seed if synthetic_seed is None else synthetic_seed
    effective_model_type = config.model_type if model_type_override is None else model_type_override
    features_df, _ = load_feature_matrix(
        price_column=config.price_column,
        features_df=features_df,
        feature_matrix_path=_config_feature_matrix_path(config),
        use_synthetic_data=_config_use_synthetic_data(config),
        synthetic_seed=effective_seed,
    )
    validate_feature_matrix_assumptions(features_df, config.price_column)

    labels = validate_label_generation(features_df, config.price_column)
    X, y, meta = build_model_dataset(features_df, labels, exclude_columns=[config.price_column, "sector_code"])

    if X.empty or y.empty:
        raise ValueError("Aligned modeling dataset is empty after feature/label assembly.")

    effective_train_end = resolve_train_end_period(features_df, labels, config.train_end_period, config.test_size)
    X_train, y_train, _, _, X_test, y_test = time_based_split(X, y, train_end_period=effective_train_end)

    if set(X_train.index).intersection(set(X_test.index)):
        raise ValueError("Invalid split: train and test datasets overlap.")

    model = ModelRegistry.create(effective_model_type)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test.copy()).rename("prediction")
    if not predictions.index.equals(X_test.index):
        raise ValueError("model.predict must preserve the exact X_test MultiIndex.")
    feature_importance = model.feature_importance(X_train.columns)
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
        feature_importance=feature_importance,
    )


def run_diagnostics(config: ResolvedConfig) -> DiagnosticsReport:
    """Run validations required by the diagnostics CLI entrypoint."""
    features_df, _ = load_feature_matrix(
        price_column=config.price_column,
        feature_matrix_path=_config_feature_matrix_path(config),
        use_synthetic_data=_config_use_synthetic_data(config),
        synthetic_seed=config.synthetic_seed,
    )
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


def summarize_run_metrics(artifacts: PipelineArtifacts) -> pd.Series:
    """Return single-run IC, RankIC, and spread metrics from pipeline artifacts."""
    decile_summary = decile_return_summary(artifacts.y_test, artifacts.predictions)
    return pd.Series(
        {
            "IC_mean": float(artifacts.ic_by_period.mean()) if not artifacts.ic_by_period.empty else float("nan"),
            "RankIC_mean": float(artifacts.rank_ic_by_period.mean()) if not artifacts.rank_ic_by_period.empty else float("nan"),
            "spread": float(decile_summary["spread"]),
        },
        dtype=float,
        name="run_metrics",
    )


def build_robustness_summary(per_run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run metrics into mean/std/min/max summary statistics by model."""
    if per_run_metrics.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
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
            ]
        )

    summary = (
        per_run_metrics.groupby("model_name", sort=True)
        .agg(
            IC_mean=("IC_mean", "mean"),
            IC_std=("IC_mean", "std"),
            IC_min=("IC_mean", "min"),
            IC_max=("IC_mean", "max"),
            RankIC_mean=("RankIC_mean", "mean"),
            RankIC_std=("RankIC_mean", "std"),
            RankIC_min=("RankIC_mean", "min"),
            RankIC_max=("RankIC_mean", "max"),
            spread_mean=("spread", "mean"),
            spread_std=("spread", "std"),
            spread_min=("spread", "min"),
            spread_max=("spread", "max"),
        )
        .reset_index()
    )
    numeric_columns = [column for column in summary.columns if column != "model_name"]
    summary.loc[:, numeric_columns] = summary.loc[:, numeric_columns].astype(float)
    return summary


def build_feature_stability_summary(per_run_feature_importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run feature importances into mean/std/rank-stability by model and feature."""
    if per_run_feature_importance.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "feature_name",
                "mean_importance",
                "std_importance",
                "mean_abs_importance",
                "rank_stability",
            ]
        )

    ranked = per_run_feature_importance.copy()
    ranked.loc[:, "importance_rank"] = ranked.groupby(["model_name", "run_id"], sort=False)["abs_importance"].rank(
        method="average",
        ascending=False,
    )

    summary = (
        ranked.groupby(["model_name", "feature_name"], sort=True)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            mean_abs_importance=("abs_importance", "mean"),
            rank_stability=("importance_rank", "std"),
        )
        .reset_index()
    )
    numeric_columns = [
        "mean_importance",
        "std_importance",
        "mean_abs_importance",
        "rank_stability",
    ]
    summary.loc[:, numeric_columns] = summary.loc[:, numeric_columns].astype(float)
    return summary.sort_values(
        by=["model_name", "mean_abs_importance", "feature_name"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def interpret_robustness_summary(summary: pd.DataFrame) -> list[RobustnessWarning]:
    """Generate warning flags for high variance or wide performance ranges."""
    warnings: list[RobustnessWarning] = []
    metric_columns = {
        "IC_mean": ("IC_mean", "IC_std", "IC_min", "IC_max"),
        "RankIC_mean": ("RankIC_mean", "RankIC_std", "RankIC_min", "RankIC_max"),
        "spread": ("spread_mean", "spread_std", "spread_min", "spread_max"),
    }

    for row in summary.to_dict(orient="records"):
        model_name = str(row["model_name"])
        for metric_name, (mean_key, std_key, min_key, max_key) in metric_columns.items():
            mean_value = float(row[mean_key])
            std_value = float(row[std_key])
            min_value = float(row[min_key])
            max_value = float(row[max_key])
            if np.isnan(std_value) or np.isnan(min_value) or np.isnan(max_value):
                continue

            range_value = max_value - min_value
            rules = HIGH_VARIANCE_RULES[metric_name]
            std_threshold = max(rules["std_floor"], abs(mean_value) * 0.75)
            range_threshold = max(rules["range_floor"], abs(mean_value) * 1.50)

            if std_value > std_threshold:
                warnings.append(
                    RobustnessWarning(
                        model_name=model_name,
                        metric_name=metric_name,
                        warning_type="high_variance",
                        message=(
                            f"WARNING: {model_name} shows high variance for {metric_name} "
                            f"(std={std_value:.6f}, threshold={std_threshold:.6f})."
                        ),
                    )
                )

            if range_value > range_threshold:
                warnings.append(
                    RobustnessWarning(
                        model_name=model_name,
                        metric_name=metric_name,
                        warning_type="instability",
                        message=(
                            f"WARNING: {model_name} shows instability for {metric_name} "
                            f"(min={min_value:.6f}, max={max_value:.6f}, range={range_value:.6f})."
                        ),
                    )
                )
    return warnings


def interpret_feature_stability(summary: pd.DataFrame) -> list[FeatureStabilityWarning]:
    """Generate interpretation flags for unstable or consistently strong features."""
    warnings: list[FeatureStabilityWarning] = []
    if summary.empty:
        return warnings

    for row in summary.to_dict(orient="records"):
        model_name = str(row["model_name"])
        feature_name = str(row["feature_name"])
        mean_importance = float(row["mean_importance"])
        std_importance = float(row["std_importance"])
        mean_abs_importance = float(row["mean_abs_importance"])
        rank_stability = float(row["rank_stability"])

        if (
            np.isfinite(std_importance)
            and np.isfinite(mean_abs_importance)
            and std_importance > max(0.0, mean_abs_importance * FEATURE_STABILITY_STD_RATIO)
        ):
            warnings.append(
                FeatureStabilityWarning(
                    model_name=model_name,
                    feature_name=feature_name,
                    warning_type="unstable_feature",
                    message=(
                        f"WARNING: {model_name} feature '{feature_name}' is unstable "
                        f"(std_importance={std_importance:.6f}, mean_abs_importance={mean_abs_importance:.6f})."
                    ),
                )
            )

        if (
            np.isfinite(mean_abs_importance)
            and mean_abs_importance > 0.0
            and np.isfinite(rank_stability)
            and rank_stability <= FEATURE_SIGNAL_RANK_STD_MAX
        ):
            warnings.append(
                FeatureStabilityWarning(
                    model_name=model_name,
                    feature_name=feature_name,
                    warning_type="consistent_signal",
                    message=(
                        f"INFO: {model_name} feature '{feature_name}' shows consistent signal "
                        f"(mean_abs_importance={mean_abs_importance:.6f}, rank_stability={rank_stability:.6f})."
                    ),
                )
            )
    return warnings


def run_robustness_analysis(
    config: ResolvedConfig,
    *,
    model_names: list[str] | None = None,
    run_count: int | None = None,
) -> RobustnessAnalysis:
    """Execute repeated seeded runs and summarize model robustness."""
    selected_model_names = model_names or ModelRegistry.available_models()
    run_total = config.robustness_runs if run_count is None else run_count
    if run_total <= 0:
        raise ValueError("run_count must be a positive integer.")

    rows: list[dict[str, float | int | str]] = []
    feature_rows: list[dict[str, float | int | str]] = []
    for run_idx in range(run_total):
        seed = config.synthetic_seed + run_idx
        for model_name in selected_model_names:
            artifacts = build_pipeline_artifacts(
                config,
                synthetic_seed=seed,
                model_type_override=model_name,
            )
            metrics = summarize_run_metrics(artifacts)
            rows.append(
                {
                    "run_id": run_idx + 1,
                    "seed": seed,
                    "model_name": model_name,
                    "IC_mean": float(metrics["IC_mean"]),
                    "RankIC_mean": float(metrics["RankIC_mean"]),
                    "spread": float(metrics["spread"]),
                }
            )
            for feature_name, importance in artifacts.feature_importance.items():
                feature_rows.append(
                    {
                        "run_id": run_idx + 1,
                        "seed": seed,
                        "model_name": model_name,
                        "feature_name": str(feature_name),
                        "importance": float(importance),
                        "abs_importance": float(abs(importance)),
                    }
                )

    per_run_metrics = pd.DataFrame(
        rows,
        columns=["run_id", "seed", "model_name", "IC_mean", "RankIC_mean", "spread"],
    )
    per_run_feature_importance = pd.DataFrame(
        feature_rows,
        columns=["run_id", "seed", "model_name", "feature_name", "importance", "abs_importance"],
    )
    summary = build_robustness_summary(per_run_metrics)
    warnings = interpret_robustness_summary(summary)
    feature_stability = build_feature_stability_summary(per_run_feature_importance)
    feature_warnings = interpret_feature_stability(feature_stability)
    return RobustnessAnalysis(
        per_run_metrics=per_run_metrics,
        summary=summary,
        warnings=warnings,
        per_run_feature_importance=per_run_feature_importance,
        feature_stability=feature_stability,
        feature_warnings=feature_warnings,
    )
