"""Execute the repository test suite programmatically and summarize the results."""

from __future__ import annotations

import traceback

import pytest


def structured_step(title: str) -> None:
    print(f"=== {title} ===")


class PytestResultCollector:
    """Collect pass/fail counts while pytest runs."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call":
            return
        if report.passed:
            self.passed += 1
        elif report.failed:
            self.failed += 1



def main() -> int:
    collector = PytestResultCollector()
    try:
        structured_step("RUN PYTEST")
        exit_code = pytest.main(["tests", "-q"], plugins=[collector])

        structured_step("TEST SUMMARY")
        print("status: success" if collector.failed == 0 and exit_code == 0 else "status: failure")
        print(f"tests_passed: {collector.passed}")
        print(f"tests_failed: {collector.failed}")
        print(f"pytest_exit_code: {exit_code}")
        return int(exit_code)
    except Exception as exc:  # noqa: BLE001 - CLI should fail fast with a clear terminal error.
        structured_step("TEST RUN FAILED")
        print("status: error")
        print(f"error_type: {type(exc).__name__}")
        print(f"error_message: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
