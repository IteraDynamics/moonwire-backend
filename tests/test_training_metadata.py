# src/ml/training_metadata.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.paths import MODELS_DIR

# Default log file location
_DEFAULT_LOG = MODELS_DIR / "training_runs.jsonl"


def _as_path(path: Optional[Union[str, Path]]) -> Path:
    if path is None:
        return _DEFAULT_LOG
    return Path(path)


def save_training_metadata(
    *,
    version: str,
    rows: int,
    origin_counts: Dict[str, int],
    label_counts: Dict[str, int],
    metrics: Dict[str, Dict[str, float]],
    top_features: list[str],
    path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Append one training-run summary JSON object to training_runs.jsonl.

    Returns the path written to.
    """
    out_path = _as_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "rows": int(rows),
        "origin_counts": dict(origin_counts or {}),
        "label_counts": dict(label_counts or {}),
        "metrics": metrics or {},
        "top_features": list(top_features or []),
    }

    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

    return out_path


def load_latest_training_metadata(
    path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Return the most recent JSON object from training_runs.jsonl (or None if empty/missing).
    """
    p = _as_path(path)
    if not p.exists():
        return None

    latest: Optional[Dict[str, Any]] = None
    try:
        # Read from the end efficiently without loading entire file in huge scenarios
        # Simpler approach here since files are small in CI: read all lines.
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            return None
        for ln in reversed(lines):
            try:
                latest = json.loads(ln)
                break
            except Exception:
                continue
    except Exception:
        return None
    return latest


__all__ = ["save_training_metadata", "load_latest_training_metadata"]