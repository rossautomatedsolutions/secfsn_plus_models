from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """Deterministic quarterly panel used across pipeline tests."""
    ciks = ["0001", "0002", "0003", "0004"]
    periods = [
        "2020Q1",
        "2020Q2",
        "2020Q3",
        "2020Q4",
        "2021Q1",
        "2021Q2",
        "2021Q3",
    ]
    index = pd.MultiIndex.from_product([ciks, periods], names=["cik", "period"])

    rows: list[dict[str, float | int]] = []
    for cik_idx, _cik in enumerate(ciks):
        base_price = 100.0 + (25.0 * cik_idx)
        for period_idx, _period in enumerate(periods):
            rows.append(
                {
                    "price": base_price + (5.0 * period_idx) + (2.0 * cik_idx * period_idx),
                    "feature_1": float(period_idx + 1),
                    "feature_2": float((period_idx + 1) ** 2),
                    "feature_3": float(cik_idx + 1),
                    "feature_4": float((cik_idx + 1) * (period_idx + 1)),
                    "feature_5": float((cik_idx - period_idx) ** 2),
                    "feature_6": float((cik_idx + 2) * 0.1 + period_idx),
                    "sector_code": cik_idx % 2,
                }
            )

    return pd.DataFrame(rows, index=index)


@pytest.fixture
def single_period_panel(synthetic_panel: pd.DataFrame) -> pd.DataFrame:
    first_period = synthetic_panel.index.get_level_values("period").unique()[0]
    return synthetic_panel.loc[pd.IndexSlice[:, first_period], :].copy()


@pytest.fixture
def manual_metric_panel() -> tuple[pd.Series, pd.Series]:
    index = pd.MultiIndex.from_tuples(
        [
            ("0001", "2020Q1"),
            ("0002", "2020Q1"),
            ("0003", "2020Q1"),
            ("0001", "2020Q2"),
            ("0002", "2020Q2"),
            ("0003", "2020Q2"),
        ],
        names=["cik", "period"],
    )
    y_true = pd.Series([1.0, 2.0, 3.0, 3.0, 2.0, 1.0], index=index, name="target")
    y_pred = pd.Series([10.0, 20.0, 30.0, 30.0, 20.0, 10.0], index=index, name="prediction")
    return y_true, y_pred
