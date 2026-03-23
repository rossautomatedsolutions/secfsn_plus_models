"""Runtime configuration for CLI automation scripts in ``secfsn_plus_models``."""

from __future__ import annotations

# Required modeling configuration.
price_column = "price"
train_end_period = "2020Q4"
model_type = "linear"

# Optional split override: when set to a positive integer and ``train_end_period`` is ``None``,
# the last ``test_size`` quarterly periods are reserved for testing.
test_size = None

# Optional real dataset path. Supported formats are CSV, Parquet, and Pickle. When this remains
# ``None`` or the file does not exist, the CLI scripts automatically generate a synthetic dataset.
feature_matrix_path = None

# Synthetic-data robustness settings. When the feature matrix falls back to the built-in synthetic
# generator, ``synthetic_seed`` is the first seed used and ``robustness_runs`` controls how many
# seeded reruns are executed for robustness analysis.
synthetic_seed = 20240323
robustness_runs = 10
