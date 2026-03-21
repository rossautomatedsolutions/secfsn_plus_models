"""Run structural diagnostics for the SECFSN+ feature matrix, labels, and splits."""

from __future__ import annotations

import traceback

import config


def structured_step(title: str) -> None:
    print(f"=== {title} ===")


def main() -> int:
    try:
        from secfsn_plus_models.automation import load_feature_matrix, resolve_runtime_config, run_diagnostics

        runtime_config = resolve_runtime_config(config)

        structured_step("LOAD FEATURE MATRIX")
        _, source_message = load_feature_matrix(runtime_config.price_column, runtime_config.feature_matrix_path)
        print("status: success")
        print(f"dataset_source: {source_message}")

        report = run_diagnostics(runtime_config)

        structured_step("FEATURE MATRIX ASSUMPTIONS")
        print("status: success")
        print(f"row_count: {report.row_count}")
        print(f"column_count: {report.column_count}")
        print(f"numeric_feature_coverage:\n{report.numeric_coverage.to_string()}")

        structured_step("LABEL GENERATION")
        print("status: success")
        print(f"label_rows: {report.label_count}")
        print(f"no_forward_leakage: {report.leakage_check_passed}")
        print(f"correct_alignment: {report.alignment_check_passed}")

        structured_step("SPLIT VALIDATION")
        print("status: success")
        print(f"train_rows: {report.train_rows}")
        print(f"test_rows: {report.test_rows}")
        print(f"overlap_rows: {report.split_overlap_count}")
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should fail fast with a clear terminal error.
        structured_step("DIAGNOSTICS FAILED")
        print("status: error")
        print(f"error_type: {type(exc).__name__}")
        print(f"error_message: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
