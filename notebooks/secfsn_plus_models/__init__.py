"""Notebook-local namespace package that points imports at the repository source tree."""

from __future__ import annotations

from pathlib import Path

_REAL_PACKAGE_DIR = Path(__file__).resolve().parents[2] / "secfsn_plus_models"

__path__ = [str(_REAL_PACKAGE_DIR)]
