"""Run a notebook-free research diagnostics snapshot for the SECFSN+ demo artifacts."""

from __future__ import annotations

import math
import traceback
from dataclasses import replace

import config


LOW_VARIANCE_THRESHOLD = 1e-6
UNUSUALLY_HIGH_IC_THRESHOLD = 0.2
LARGE_SPREAD_VARIANCE_THRESHOLD = 0.03
ROBUSTNESS_SEEDS = list(range(1, 11))
COMPARISON_MODELS = ["linear_regression", "random_forest"]
COMPARISON_COLUMNS = [
    "model_name",
    "IC_mean",
    "RankIC_mean",
    "prediction_mean",
    "prediction_std",
    "top_decile_return",
    "bottom_decile_return",
    "spread",
]
ROBUSTNESS_DISPLAY_COLUMNS = [
    "model_name",
    "IC_mean_avg",
    "IC_std",
    "IC_min",
    "IC_max",
    "RankIC_avg",
    "spread_avg",
]


def structured_step(title: str) -> None:
    """Print a consistently formatted section header."""
    print(f"=== {title} ===")


def _format_scalar(value: object) -> str:
    """Render scalar values consistently for terminal output."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6f}"
    return str(value)


def _print_kv(key: str, value: object) -> None:
    """Print a single key/value line."""
    print(f"{key}: {_format_scalar(value)}")


def _series_stats(series):
    """Return a consistent scalar stats mapping for a numeric series."""
    cleaned = series.dropna()
    if cleaned.empty:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(cleaned.mean()),
        "std": float(cleaned.std(ddof=1)),
        "min": float(cleaned.min()),
        "max": float(cleaned.max()),
    }


def _build_decile_table(y_true, y_pred):
    """Rank predictions into global deciles and summarize average realized return."""
    import pandas as pd

    frame = pd.DataFrame({"actual_return": y_true, "prediction": y_pred}).dropna()
    if frame.empty:
        return frame

    ranked_predictions = frame["prediction"].rank(method="first")
    frame["decile"] = pd.qcut(ranked_predictions, q=10, labels=range(1, 11), duplicates="drop")
    frame = frame.dropna(subset=["decile"]).copy()
    frame["decile"] = frame["decile"].astype(int)

    return (
        frame.groupby("decile", sort=True)
        .agg(
            observation_count=("actual_return", "size"),
            average_return=("actual_return", "mean"),
            average_prediction=("prediction", "mean"),
        )
        .sort_index()
    )


def _load_model_demos(runtime_config):
    from secfsn_plus_models.notebook_utils import load_notebook_demo_artifacts

    model_names = []
    for model_name in [runtime_config.model_type, *COMPARISON_MODELS]:
        if model_name not in model_names:
            model_names.append(model_name)

    demos = {}
    for model_name in model_names:
        demos[model_name] = load_notebook_demo_artifacts(
            price_column=runtime_config.price_column,
            train_end_period=runtime_config.train_end_period,
            model_type=model_name,
            test_size=runtime_config.test_size,
            feature_matrix_path=runtime_config.feature_matrix_path,
        )
    return demos


def _build_model_summary(model_name, demo):
    from secfsn_plus_models.notebook_utils import summarize_model_comparison_metrics

    summary = summarize_model_comparison_metrics(
        model_name,
        demo.pipeline.y_test,
        demo.pipeline.predictions,
        ic_by_period=demo.pipeline.ic_by_period,
        rank_ic_by_period=demo.pipeline.rank_ic_by_period,
    )
    return summary.reindex(COMPARISON_COLUMNS)


def _print_model_section(model_name, demo) -> None:
    import pandas as pd

    structured_step(f"MODEL PERFORMANCE [{model_name}]")
    summary = _build_model_summary(model_name, demo)
    for key in COMPARISON_COLUMNS[1:]:
        _print_kv(key, summary[key])
    print("IC_by_period:")
    print(demo.pipeline.ic_by_period.to_string() if not demo.pipeline.ic_by_period.empty else "no valid IC values")
    print("RankIC_by_period:")
    print(
        demo.pipeline.rank_ic_by_period.to_string()
        if not demo.pipeline.rank_ic_by_period.empty
        else "no valid RankIC values"
    )
    print()

    structured_step(f"DECILE ANALYSIS [{model_name}]")
    decile_table = _build_decile_table(demo.pipeline.y_test, demo.pipeline.predictions)
    print("decile_table:")
    print(decile_table.to_string() if not decile_table.empty else "no valid decile rows")
    _print_kv("top_decile_return", summary["top_decile_return"])
    _print_kv("bottom_decile_return", summary["bottom_decile_return"])
    _print_kv("spread", summary["spread"])
    print()

    structured_step(f"PREDICTION STATS [{model_name}]")
    prediction_stats = _series_stats(demo.pipeline.predictions)
    _print_kv("prediction_mean", prediction_stats["mean"])
    _print_kv("prediction_std", prediction_stats["std"])
    _print_kv("prediction_min", prediction_stats["min"])
    _print_kv("prediction_max", prediction_stats["max"])
    print()

    structured_step(f"SANITY FLAGS [{model_name}]")
    warnings: list[str] = []
    if pd.notna(summary["IC_mean"]) and float(summary["IC_mean"]) > UNUSUALLY_HIGH_IC_THRESHOLD:
        warnings.append("WARNING: unusually high IC")

    label_stats = _series_stats(demo.y)
    if pd.notna(label_stats["std"]) and label_stats["std"] < LOW_VARIANCE_THRESHOLD:
        warnings.append("WARNING: low variance")

    nan_checks = {
        "X_has_nans": bool(demo.X.isna().any().any()),
        "y_has_nans": bool(demo.y.isna().any()),
        "predictions_have_nans": bool(demo.pipeline.predictions.isna().any()),
    }
    for key, value in nan_checks.items():
        _print_kv(key, value)
    if any(nan_checks.values()):
        warnings.append("WARNING: NaNs detected")

    if warnings:
        for warning in warnings:
            print(warning)
    else:
        print("No sanity warnings triggered.")
    print()


def _run_seeded_robustness_analysis(runtime_config, model_names):
    from secfsn_plus_models.automation import build_robustness_summary, run_robustness_analysis

    robustness_config = replace(
        runtime_config,
        feature_matrix_path=None,
        synthetic_seed=ROBUSTNESS_SEEDS[0],
        robustness_runs=len(ROBUSTNESS_SEEDS),
    )
    robustness = run_robustness_analysis(
        robustness_config,
        model_names=model_names,
        run_count=len(ROBUSTNESS_SEEDS),
    )
    per_run_metrics = robustness.per_run_metrics.loc[
        robustness.per_run_metrics["seed"].isin(ROBUSTNESS_SEEDS)
    ].copy()
    per_run_metrics.loc[:, "run_id"] = per_run_metrics["seed"]
    summary = build_robustness_summary(per_run_metrics).rename(
        columns={
            "IC_mean": "IC_mean_avg",
            "RankIC_mean": "RankIC_mean_avg",
            "spread_mean": "spread_avg",
        }
    )
    summary = summary.rename(columns={"RankIC_mean_avg": "RankIC_avg"})
    return per_run_metrics, summary


def _build_robustness_flags(summary):
    import pandas as pd

    warnings: list[str] = []
    for row in summary.to_dict(orient="records"):
        model_name = str(row["model_name"])
        ic_mean_avg = float(row["IC_mean_avg"])
        ic_std = float(row["IC_std"])
        ic_min = float(row["IC_min"])
        spread_avg = float(row["spread_avg"])
        spread_std = float(row["spread_std"])

        if pd.notna(ic_std) and pd.notna(ic_mean_avg) and ic_std > ic_mean_avg:
            warnings.append(f"WARNING: {model_name} unstable signal (IC_std > IC_mean_avg).")
        if pd.notna(ic_min) and ic_min < 0:
            warnings.append(f"WARNING: {model_name} inconsistent direction (IC_min < 0).")
        if pd.notna(spread_std) and spread_std > max(abs(spread_avg), LARGE_SPREAD_VARIANCE_THRESHOLD):
            warnings.append(f"WARNING: {model_name} unstable ranking (large spread variance).")
    return warnings


def _print_robustness_section(runtime_config, model_names) -> None:
    _, summary = _run_seeded_robustness_analysis(runtime_config, model_names)

    structured_step("ROBUSTNESS ANALYSIS")
    _print_kv("dataset_source", "regenerated synthetic dataset for each seed")
    _print_kv("seeds", ", ".join(str(seed) for seed in ROBUSTNESS_SEEDS))
    print(
        summary.loc[:, ROBUSTNESS_DISPLAY_COLUMNS].to_string(
            index=False,
            formatters={column: _format_scalar for column in ROBUSTNESS_DISPLAY_COLUMNS[1:]},
        )
    )
    print()

    structured_step("ROBUSTNESS FLAGS")
    warnings = _build_robustness_flags(summary)
    if warnings:
        for warning in warnings:
            print(warning)
    else:
        print("No robustness warnings triggered.")
    print()


def main() -> int:
    try:
        import pandas as pd

        from secfsn_plus_models.automation import resolve_runtime_config

        runtime_config = resolve_runtime_config(config)
        demos = _load_model_demos(runtime_config)
        baseline_demo = demos[runtime_config.model_type]

        structured_step("LOAD DATA")
        _print_kv("status", "success")
        _print_kv("dataset_source", baseline_demo.dataset_source)
        _print_kv("models_evaluated", ", ".join(demos))
        _print_kv("X_shape", baseline_demo.X.shape)
        _print_kv("y_rows", len(baseline_demo.y))
        _print_kv("prediction_rows", len(baseline_demo.pipeline.predictions))
        print()

        structured_step("DATASET SUMMARY")
        _print_kv("row_count", len(baseline_demo.X))
        _print_kv("feature_count", len(baseline_demo.X.columns))
        _print_kv("unique_cik_count", int(baseline_demo.X.index.get_level_values("cik").nunique()))
        _print_kv("unique_period_count", int(baseline_demo.X.index.get_level_values("period").nunique()))
        print()

        structured_step("LABEL STATS")
        label_stats = _series_stats(baseline_demo.y)
        for key, value in label_stats.items():
            _print_kv(key, value)
        print()

        comparison_rows = []
        for model_name, demo in demos.items():
            _print_model_section(model_name, demo)
            comparison_rows.append(_build_model_summary(model_name, demo).to_dict())

        structured_step("MODEL COMPARISON")
        comparison = pd.DataFrame(comparison_rows, columns=COMPARISON_COLUMNS)
        print(
            comparison.to_string(
                index=False,
                formatters={column: _format_scalar for column in COMPARISON_COLUMNS[1:]},
            )
        )
        print()

        _print_robustness_section(runtime_config, list(demos))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should fail fast with a clear terminal error.
        structured_step("RESEARCH DIAGNOSTICS FAILED")
        print("status: error")
        print(f"error_type: {type(exc).__name__}")
        print(f"error_message: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
