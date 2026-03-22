from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_bootstrap_adds_repo_root_and_notebook_utils_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = "\n".join(
        [
            "from secfsn_plus_models.bootstrap import bootstrap",
            "bootstrap()",
            "from secfsn_plus_models.notebook_utils import (",
            "    load_notebook_demo_artifacts,",
            "    preview_aligned_dataset,",
            "    summarize_dataset,",
            "    summarize_period_coverage,",
            ")",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        cwd=repo_root / "notebooks",
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
