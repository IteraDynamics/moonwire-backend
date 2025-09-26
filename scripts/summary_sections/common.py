# scripts/summary_sections/common.py
"""
Common types and helpers for summary sections.

Exports:
- SummaryContext: lightweight context with sane defaults
- ensure_dir(path): mkdir -p
- _iso(dt): UTC ISO-8601 helper (Z)
- _load_jsonl(path): read JSONL -> list[dict]
- _write_json(path, obj): write compact JSON with sorted keys
- _write_jsonl(path, rows): append (or write) rows to JSONL
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


# ----------------------------
# Compatibility helper funcs
# ----------------------------

def ensure_dir(p: Path | str) -> Path:
    """Create directory if missing; return Path."""
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _iso(dt: datetime) -> str:
    """Format datetime in UTC ISO-8601 with trailing Z."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts; empty list if missing."""
    fp = Path(path)
    if not fp.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path | str, obj: Any) -> None:
    """Write JSON with sorted keys and UTF-8 encoding."""
    fp = Path(path)
    ensure_dir(fp.parent)
    fp.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path | str, rows: Iterable[Dict[str, Any]], mode: str = "a") -> None:
    """
    Append/write iterable of dict rows as JSONL.
    mode='a' to append, mode='w' to overwrite.
    """
    fp = Path(path)
    ensure_dir(fp.parent)
    with fp.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False))
            f.write("\n")


# ----------------------------
# Context object
# ----------------------------

@dataclass
class SummaryContext:
    """
    Lightweight context passed into summary-section appenders.

    Required:
      - logs_dir:    directory containing append-only logs
      - models_dir:  directory for JSON artifacts
      - is_demo:     bool flag toggling demo/seeded behavior

    Backwards-compatible defaults:
      - artifacts_dir: defaults to ./artifacts if not provided
    """
    logs_dir: Path
    models_dir: Path
    is_demo: bool

    # Default artifacts directory for callers/tests that don't pass it
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))

    # Optional fields used by some sections
    origins_rows: List[Dict[str, Any]] = field(default_factory=list)
    yield_data: Any = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    caches: Dict[str, Any] = field(default_factory=dict)

    # Convenience helpers
    def ensure_dirs(self) -> None:
        """Create directories if they don't exist."""
        ensure_dir(self.logs_dir)
        ensure_dir(self.models_dir)
        ensure_dir(self.artifacts_dir)

    def model_path(self, name: str) -> Path:
        """Path under models_dir."""
        return Path(self.models_dir) / name

    def artifact_path(self, name: str) -> Path:
        """Path under artifacts_dir."""
        return Path(self.artifacts_dir) / name


__all__ = [
    "SummaryContext",
    "ensure_dir",
    "_iso",
    "_load_jsonl",
    "_write_json",
    "_write_jsonl",
]
