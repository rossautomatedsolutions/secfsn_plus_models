"""Run the full SECFSN+ modeling pipeline with no manual wiring required."""

from __future__ import annotations

import math
import traceback

import config


def structured_step(title: str) -> None:
    print(f"=== {title} ===")


def _format_scalar(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6f}"
    return str(value)


def main() -> int:
    try:
        from secfsn_plus_models.automation import (
            build_pipeline_artifacts,
            load_feature_matrix,
            resolve_runtime_config,
            run_robustness_analysis,
            summarize_run_metrics,
        )

        runtime_config = resolve_runtime_config(config)

        structured_step("LOAD FEATURE MATRIX")
        _, source_message = load_feature_matrix(
            runtime_config.price_column,
            runtime_config.feature_matrix_path,
            synthetic_seed=runtime_config.synthetic_seed,
        )
        print(f"status: success")
        print(f"dataset_source: {source_message}")

        artifacts = build_pipeline_artifacts(runtime_config)

        structured_step("GENERATE LABELS")
        print("status: success")
        print(f"label_name: {artifacts.labels.name}")
        print(f"label_rows: {len(artifacts.labels)}")

        structured_step("BUILD DATASET")
        print("status: success")
        print(f"X_shape: {artifacts.X.shape}")
        print(f"y_rows: {len(artifacts.y)}")
        print(f"meta_columns: {list(artifacts.meta.columns)}")

        structured_step("TIME-BASED SPLIT")
        print("status: success")
        print(f"train_rows: {len(artifacts.X_train)}")
        print(f"test_rows: {len(artifacts.X_test)}")
        print(f"train_periods: {sorted(artifacts.X_train.index.get_level_values('period').unique().tolist())}")
        print(f"test_periods: {sorted(artifacts.X_test.index.get_level_values('period').unique().tolist())}")

        structured_step("TRAIN MODEL")
        print("status: success")
        print(f"model_type: {runtime_config.model_type}")
        print(f"train_columns: {list(artifacts.X_train.columns)}")

        structured_step("GENERATE PREDICTIONS")
        print("status: success")
        print(f"prediction_rows: {len(artifacts.predictions)}")
        print(f"prediction_preview:\n{artifacts.predictions.head().to_string()}")

        structured_step("COMPUTE METRICS")
        run_metrics = summarize_run_metrics(artifacts)
        print("status: success")
        print(f"IC_mean: {_format_scalar(float(run_metrics['IC_mean']))}")
        print(f"RankIC_mean: {_format_scalar(float(run_metrics['RankIC_mean']))}")
        print(f"spread: {_format_scalar(float(run_metrics['spread']))}")
        print(f"IC_by_period:\n{artifacts.ic_by_period.to_string() if not artifacts.ic_by_period.empty else 'no valid IC values'}")
        print(
            "RankIC_by_period:\n"
            f"{artifacts.rank_ic_by_period.to_string() if not artifacts.rank_ic_by_period.empty else 'no valid RankIC values'}"
        )

        robustness = run_robustness_analysis(runtime_config)

        structured_step("ROBUSTNESS RUN METRICS")
        print(
            robustness.per_run_metrics.to_string(
                index=False,
                formatters={
                    "IC_mean": _format_scalar,
                    "RankIC_mean": _format_scalar,
                    "spread": _format_scalar,
                },
            )
        )

        structured_step("ROBUSTNESS ANALYSIS")
        print(
            robustness.summary.to_string(
                index=False,
                formatters={
                    "IC_mean": _format_scalar,
                    "IC_std": _format_scalar,
                    "IC_min": _format_scalar,
                    "IC_max": _format_scalar,
                    "RankIC_mean": _format_scalar,
                    "RankIC_std": _format_scalar,
                    "RankIC_min": _format_scalar,
                    "RankIC_max": _format_scalar,
                    "spread_mean": _format_scalar,
                    "spread_std": _format_scalar,
                    "spread_min": _format_scalar,
                    "spread_max": _format_scalar,
                },
            )
        )

        structured_step("ROBUSTNESS FLAGS")
        if robustness.warnings:
            for warning in robustness.warnings:
                print(warning.message)
        else:
            print("No robustness warnings triggered.")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should fail fast with a clear terminal error.
        structured_step("PIPELINE FAILED")
        print(f"status: error")
        print(f"error_type: {type(exc).__name__}")
        print(f"error_message: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
