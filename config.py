"""Runtime configuration for CLI automation scripts in ``secfsn_plus_models``."""

from __future__ import annotations

# Required modeling configuration.
price_column = "price"
train_end_period = "2020Q4"
model_type = "linear"

# Optional split override: when set to a positive integer and ``train_end_period`` is ``None``,
# the last ``test_size`` quarterly periods are reserved for testing.
test_size = None

# Data source controls.
USE_SYNTHETIC_DATA = False
DATA_PATH = "path_to_secfsn_plus_output.parquet"

# Backward-compatible aliases.
use_synthetic_data = USE_SYNTHETIC_DATA
data_path = DATA_PATH
feature_matrix_path = DATA_PATH

# Synthetic-data robustness settings. When the feature matrix falls back to the built-in synthetic
# generator, ``synthetic_seed`` is the first seed used and ``robustness_runs`` controls how many
# seeded reruns are executed for robustness analysis.
synthetic_seed = 20240323
robustness_runs = 10
