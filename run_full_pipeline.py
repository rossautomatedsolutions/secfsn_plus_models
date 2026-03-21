"""Run the full SECFSN+ modeling pipeline with no manual wiring required."""

from __future__ import annotations

import traceback

import config


def structured_step(title: str) -> None:
    print(f"=== {title} ===")


def main() -> int:
    try:
        from secfsn_plus_models.automation import build_pipeline_artifacts, load_feature_matrix, resolve_runtime_config

        runtime_config = resolve_runtime_config(config)

        structured_step("LOAD FEATURE MATRIX")
        _, source_message = load_feature_matrix(runtime_config.price_column, runtime_config.feature_matrix_path)
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
        print("status: success")
        print(f"IC_mean: {float(artifacts.ic_by_period.mean()) if not artifacts.ic_by_period.empty else float('nan'):.6f}")
        print(
            f"RankIC_mean: {float(artifacts.rank_ic_by_period.mean()) if not artifacts.rank_ic_by_period.empty else float('nan'):.6f}"
        )
        print(f"IC_by_period:\n{artifacts.ic_by_period.to_string() if not artifacts.ic_by_period.empty else 'no valid IC values'}")
        print(
            "RankIC_by_period:\n"
            f"{artifacts.rank_ic_by_period.to_string() if not artifacts.rank_ic_by_period.empty else 'no valid RankIC values'}"
        )
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
