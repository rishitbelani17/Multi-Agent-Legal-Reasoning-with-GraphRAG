"""
General-purpose utilities.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a readable format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def ensure_dir(path: str) -> str:
    """Create directory (and parents) if it doesn't exist. Return path."""
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
    logging.getLogger(__name__).info("Saved → %s", path)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy types etc."""
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)
