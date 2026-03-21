from __future__ import annotations

import pandas as pd

from secfsn_plus_models.dataset import build_model_dataset
from secfsn_plus_models.labels import compute_next_period_return


def test_build_model_dataset_aligns_X_and_y_and_preserves_metadata(synthetic_panel: pd.DataFrame) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")

    X, y, meta = build_model_dataset(
        synthetic_panel,
        labels,
        exclude_columns=["price", "sector_code"],
    )

    assert X.index.equals(y.index)
    assert meta.index.equals(y.index)
    assert X.shape[0] == y.shape[0] == meta.shape[0]
    assert list(meta.columns) == ["price", "sector_code"]
    assert "price" not in X.columns
    assert "sector_code" not in X.columns
    assert y.name == "next_period_return"


def test_build_model_dataset_removes_rows_without_labels(synthetic_panel: pd.DataFrame) -> None:
    labels = compute_next_period_return(synthetic_panel, "price")

    X, y, meta = build_model_dataset(synthetic_panel, labels, exclude_columns=["price"])

    assert len(X) == len(labels)
    assert X.index.equals(labels.index)
    assert y.index.equals(labels.index)
    assert meta.index.equals(labels.index)
    assert "2021Q3" not in X.index.get_level_values("period")
