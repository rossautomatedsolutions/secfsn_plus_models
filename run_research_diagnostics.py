"""Run a notebook-free research diagnostics snapshot for the SECFSN+ demo artifacts."""

from __future__ import annotations

import math
import traceback

import config


LOW_VARIANCE_THRESHOLD = 1e-6
UNUSUALLY_HIGH_IC_THRESHOLD = 0.2


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
        return frame, float("nan"), float("nan"), float("nan")

    ranked_predictions = frame["prediction"].rank(method="first")
    frame["decile"] = pd.qcut(ranked_predictions, q=10, labels=range(1, 11), duplicates="drop")
    frame = frame.dropna(subset=["decile"]).copy()
    frame["decile"] = frame["decile"].astype(int)

    decile_table = (
        frame.groupby("decile", sort=True)
        .agg(
            observation_count=("actual_return", "size"),
            average_return=("actual_return", "mean"),
            average_prediction=("prediction", "mean"),
        )
        .sort_index()
    )

    if decile_table.empty:
        return decile_table, float("nan"), float("nan"), float("nan")

    top_decile = float(decile_table.iloc[-1]["average_return"])
    bottom_decile = float(decile_table.iloc[0]["average_return"])
    spread = top_decile - bottom_decile
    return decile_table, top_decile, bottom_decile, spread


def main() -> int:
    try:
        import pandas as pd

        from secfsn_plus_models.automation import resolve_runtime_config
        from secfsn_plus_models.notebook_utils import load_notebook_demo_artifacts

        runtime_config = resolve_runtime_config(config)

        structured_step("LOAD DATA")
        demo = load_notebook_demo_artifacts(
            price_column=runtime_config.price_column,
            train_end_period=runtime_config.train_end_period,
            model_type=runtime_config.model_type,
            test_size=runtime_config.test_size,
            feature_matrix_path=runtime_config.feature_matrix_path,
        )
        X = demo.X
        y = demo.y
        predictions = demo.pipeline.predictions
        y_test = demo.pipeline.y_test

        _print_kv("status", "success")
        _print_kv("dataset_source", demo.dataset_source)
        _print_kv("X_shape", X.shape)
        _print_kv("y_rows", len(y))
        _print_kv("prediction_rows", len(predictions))
        print()

        structured_step("DATASET SUMMARY")
        _print_kv("row_count", len(X))
        _print_kv("feature_count", len(X.columns))
        _print_kv("unique_cik_count", int(X.index.get_level_values("cik").nunique()))
        _print_kv("unique_period_count", int(X.index.get_level_values("period").nunique()))
        print()

        structured_step("LABEL STATS")
        label_stats = _series_stats(y)
        for key, value in label_stats.items():
            _print_kv(key, value)
        print()

        structured_step("MODEL PERFORMANCE")
        ic_by_period = demo.pipeline.ic_by_period
        rank_ic_by_period = demo.pipeline.rank_ic_by_period
        ic_mean = float(ic_by_period.mean()) if not ic_by_period.empty else float("nan")
        rank_ic_mean = float(rank_ic_by_period.mean()) if not rank_ic_by_period.empty else float("nan")
        _print_kv("IC_mean", ic_mean)
        _print_kv("RankIC_mean", rank_ic_mean)
        print("IC_by_period:")
        print(ic_by_period.to_string() if not ic_by_period.empty else "no valid IC values")
        print("RankIC_by_period:")
        print(rank_ic_by_period.to_string() if not rank_ic_by_period.empty else "no valid RankIC values")
        print()

        structured_step("DECILE ANALYSIS")
        decile_table, top_decile_return, bottom_decile_return, spread = _build_decile_table(y_test, predictions)
        print("decile_table:")
        print(decile_table.to_string() if not decile_table.empty else "no valid decile rows")
        _print_kv("top_decile_return", top_decile_return)
        _print_kv("bottom_decile_return", bottom_decile_return)
        _print_kv("spread", spread)
        print()

        structured_step("PREDICTION STATS")
        prediction_stats = _series_stats(predictions)
        _print_kv("prediction_mean", prediction_stats["mean"])
        _print_kv("prediction_std", prediction_stats["std"])
        _print_kv("prediction_min", prediction_stats["min"])
        _print_kv("prediction_max", prediction_stats["max"])
        print()

        structured_step("SANITY FLAGS")
        warnings: list[str] = []
        if pd.notna(ic_mean) and ic_mean > UNUSUALLY_HIGH_IC_THRESHOLD:
            warnings.append("WARNING: unusually high IC")
        if pd.notna(label_stats["std"]) and label_stats["std"] < LOW_VARIANCE_THRESHOLD:
            warnings.append("WARNING: low variance")

        nan_checks = {
            "X_has_nans": bool(X.isna().any().any()),
            "y_has_nans": bool(y.isna().any()),
            "predictions_have_nans": bool(predictions.isna().any()),
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
