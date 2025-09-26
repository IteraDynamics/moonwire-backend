# scripts/summary_sections/common.py
"""
Common types and helpers for summary sections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SummaryContext:
    """
    Lightweight context passed into summary-section appenders.

    Required:
      - logs_dir:    directory containing append-only logs
      - models_dir:  directory for JSON artifacts
      - is_demo:     bool flag toggling demo/seeded behavior

    Backwards-compatible defaults:
      - artifacts_dir: if not provided by the caller, defaults to ./artifacts
                       (many sections save PNGs here)
    """
    logs_dir: Path
    models_dir: Path
    is_demo: bool

    # NEW: default artifacts directory for callers/tests that don't pass it
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))

    # Optional, used by some sections; kept for compatibility
    origins_rows: List[Dict[str, Any]] = field(default_factory=list)
    yield_data: Any = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    caches: Dict[str, Any] = field(default_factory=dict)

    # Convenience helpers
    def ensure_dirs(self) -> None:
        """Create directories if they don't exist."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def model_path(self, name: str) -> Path:
        """Path under models_dir."""
        return Path(self.models_dir) / name

    def artifact_path(self, name: str) -> Path:
        """Path under artifacts_dir."""
        return Path(self.artifacts_dir) / name
