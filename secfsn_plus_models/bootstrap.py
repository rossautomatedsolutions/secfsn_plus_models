"""Helpers for making repository-local imports notebook-friendly."""

from __future__ import annotations


def bootstrap() -> None:
    """Ensure the repository root is on ``sys.path`` for notebook imports."""
    import sys
    from pathlib import Path

    try:
        root = Path(__file__).resolve().parents[1]
    except Exception as exc:  # pragma: no cover - defensive path resolution guard
        raise RuntimeError("Failed to resolve the repository root for notebook imports.") from exc

    if not root.is_dir():
        raise RuntimeError(f"Failed to locate the repository root for notebook imports: {root}")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if root_str not in sys.path:
        raise RuntimeError(f"Failed to add the repository root to sys.path: {root}")
