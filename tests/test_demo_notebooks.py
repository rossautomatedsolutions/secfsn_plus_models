from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATHS = [
    Path("notebooks/01_dataset_inspection.ipynb"),
    Path("notebooks/02_baseline_model_training.ipynb"),
    Path("notebooks/03_model_comparison.ipynb"),
    Path("notebooks/04_prediction_evaluation.ipynb"),
    Path("notebooks/04_layer04_models_demo.ipynb"),
]

EXPECTED_FIRST_CELL = (
    "from secfsn_plus_models.notebook_entry import init_notebook\n"
    "init_notebook()\n"
)


def test_demo_notebooks_exist_and_reference_repository_utilities() -> None:
    for notebook_path in NOTEBOOK_PATHS:
        assert notebook_path.exists()
        notebook = json.loads(notebook_path.read_text())
        assert notebook["nbformat"] == 4
        assert notebook["cells"][0]["cell_type"] == "code"
        assert "".join(notebook["cells"][0]["source"]) == EXPECTED_FIRST_CELL
        code_sources = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert any("secfsn_plus_models.notebook_utils" in source for source in code_sources)
        if notebook_path.name == "04_layer04_models_demo.ipynb":
            assert any("build_model_output_frame" in source for source in code_sources)
