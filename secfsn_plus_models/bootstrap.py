"""Helpers for making repository-local imports notebook-friendly."""

from __future__ import annotations


def bootstrap() -> None:
    """
    Ensure repository root is on ``sys.path`` so notebooks can import the package.
    Safe to call multiple times.
    """
    import sys
    from pathlib import Path

    # Current file: secfsn_plus_models/bootstrap.py
    # Repo root is one level above package directory
    root = Path(__file__).resolve().parents[1]

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
