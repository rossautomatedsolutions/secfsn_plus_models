from __future__ import annotations

import pandas as pd
import pytest

from secfsn_plus_models.data_loader import load_feature_matrix


def test_load_feature_matrix_csv_with_cik_period_columns(tmp_path):
    frame = pd.DataFrame(
        [
            {"cik": 1002, "period": "2025Q2", "feature_a": 2.0},
            {"cik": 1001, "period": "2025Q3", "feature_a": 3.0},
            {"cik": 1001, "period": "2025Q1", "feature_a": 1.0},
            {"cik": 1001, "period": "2025Q2", "feature_a": 2.0},
        ]
    )
    path = tmp_path / "feature_matrix.csv"
    frame.to_csv(path, index=False)

    loaded = load_feature_matrix(str(path))

    assert list(loaded.index.names) == ["cik", "period"]
    assert loaded.index.tolist() == [
        ("1001", "2025Q1"),
        ("1001", "2025Q2"),
        ("1001", "2025Q3"),
        ("1002", "2025Q2"),
    ]
    assert loaded.loc[("1001", "2025Q1"), "feature_a"] == pytest.approx(1.0)


def test_load_feature_matrix_rejects_invalid_period_strings(tmp_path):
    frame = pd.DataFrame(
        [
            {"cik": 1001, "period": "2025-Q1", "feature_a": 1.0},
            {"cik": 1001, "period": "2025Q2", "feature_a": 2.0},
        ]
    )
    path = tmp_path / "feature_matrix.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="canonical 'YYYYQ#'"):
        load_feature_matrix(str(path))


def test_load_feature_matrix_rejects_duplicate_index_rows(tmp_path):
    frame = pd.DataFrame(
        [
            {"cik": "1001", "period": "2025Q1", "feature_a": 1.0},
            {"cik": "1001", "period": "2025Q1", "feature_a": 2.0},
        ]
    )
    path = tmp_path / "feature_matrix.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate index rows"):
        load_feature_matrix(str(path))


def test_load_feature_matrix_parquet_with_multiindex(tmp_path):
    index = pd.MultiIndex.from_tuples(
        [("1002", "2025Q2"), ("1002", "2025Q1"), ("1001", "2025Q4")],
        names=["cik", "period"],
    )
    frame = pd.DataFrame({"feature_a": [2.0, 1.0, 4.0]}, index=index)
    path = tmp_path / "feature_matrix.parquet"
    frame.to_parquet(path)

    loaded = load_feature_matrix(str(path))

    assert loaded.index.tolist() == [
        ("1001", "2025Q4"),
        ("1002", "2025Q1"),
        ("1002", "2025Q2"),
    ]


def test_load_feature_matrix_rejects_unsupported_suffix(tmp_path):
    path = tmp_path / "feature_matrix.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported feature matrix format"):
        load_feature_matrix(str(path))
