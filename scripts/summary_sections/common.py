# scripts/summary_sections/common.py
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

# ---------------------------
# Paths & filesystem helpers
# ---------------------------

def ensure_dir(p: Union[str, Path]) -> Path:
    """Create directory if missing and return it as Path."""
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d

def _read_text(path: Union[str, Path], default: str = "") -> str:
    p = Path(path)
    if not p.exists():
        return default
    return p.read_text(encoding="utf-8")

def _write_text(path: Union[str, Path], text: str) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")

def _read_json(path: Union[str, Path], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json(path: Union[str, Path], obj: Any, pretty: bool = True) -> None:
    ensure_dir(Path(path).parent)
    if pretty:
        s = json.dumps(obj, indent=2, sort_keys=False)
    else:
        s = json.dumps(obj, separators=(",", ":"), sort_keys=False)
    Path(path).write_text(s, encoding="utf-8")

def _load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts. Missing file -> empty list."""
    p = Path(path)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        except Exception:
            # Skip malformed lines
            pass
    return out

def _write_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]]) -> None:
    """Append rows to JSONL file (creating parent dir)."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            try:
                f.write(json.dumps(r, separators=(",", ":"), sort_keys=False) + "\n")
            except Exception:
                # Skip un-serializable rows
                continue

# ---------------------------
# Time utilities
# ---------------------------

ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

def _iso(dt: datetime) -> str:
    """Return ISO-8601 Z string for an aware datetime in UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def parse_ts(x: Union[str, int, float, datetime]) -> datetime:
    """
    Parse a timestamp into an aware UTC datetime.

    Accepts:
      - ISO-8601 strings (with/without 'Z')
      - epoch seconds (int/float)
      - epoch milliseconds (heuristic or fallback)
      - datetime (naive -> assume UTC; aware -> convert to UTC)
    """
    if isinstance(x, datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=timezone.utc)
        return x.astimezone(timezone.utc)

    if isinstance(x, (int, float)):
        val = float(x)
        try:
            dt = datetime.fromtimestamp(val, tz=timezone.utc)
            if dt.year > 3000:
                dt = datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)
            return dt
        except (OSError, OverflowError, ValueError):
            return datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)

    if isinstance(x, str):
        s = x.strip()
        if s.isdigit() or re.fullmatch(r"\d+(\.\d+)?", s):
            try:
                val = float(s)
                dt = datetime.fromtimestamp(val, tz=timezone.utc)
                if dt.year > 3000:
                    dt = datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)
                return dt
            except (OSError, OverflowError, ValueError):
                return datetime.fromtimestamp(val / 1000.0, tz=timezone.utc)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
        except Exception:
            if ISO_RE.match(x):
                base = x.split(".")[0]
                if base.endswith("Z"):
                    base = base[:-1]
                dt = datetime.fromisoformat(base)
            else:
                raise
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    raise TypeError(f"Unsupported timestamp type: {type(x)}")

# ---------------------------
# Env + demo helpers
# ---------------------------

def is_demo_mode() -> bool:
    """Read demo mode from environment."""
    v = os.getenv("DEMO_MODE") or os.getenv("MW_DEMO") or ""
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

def generate_demo_yield_plan_if_needed(existing: Optional[List[dict]] = None):
    """
    Lightweight stub for sections that optionally request a demo 'yield plan'.
    - If DEMO_MODE is off: return (existing or [], [])
    - If DEMO_MODE is on and no existing plan: return a tiny synthetic plan + one demo event
    The exact structure is intentionally loose; sections should tolerate empty results.
    """
    plan = list(existing or [])
    events: List[dict] = []
    if is_demo_mode() and not plan:
        # minimal, deterministic placeholders
        now = datetime.now(timezone.utc)
        plan = [
            {"origin": "reddit", "weight": 0.6, "ts": _iso(now)},
            {"origin": "twitter", "weight": 0.5, "ts": _iso(now)},
        ]
        events.append({
            "type": "demo_yield_plan_seeded",
            "at": _iso(now),
            "meta": {"version": "v0.0.1"}
        })
    return plan, events

# ---------------------------
# Small presentation helpers
# ---------------------------

def red(s: str) -> str:
    """Lightweight 'red' marker for plain-text markdown contexts."""
    return f"**{s}**"

def weight_to_label(w: float) -> str:
    """Map a weight/score [0,1] to a coarse label used in some summaries."""
    try:
        w = float(w)
    except Exception:
        return "unknown"
    if w >= 0.85:
        return "very high"
    if w >= 0.70:
        return "high"
    if w >= 0.55:
        return "medium"
    if w >= 0.40:
        return "low"
    return "very low"

# ---------------------------
# Rolling / stats helpers
# ---------------------------

def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    if not xs:
        return float("nan")
    return sum(xs) / float(len(xs))

def stdev(xs: Iterable[float]) -> float:
    xs = list(xs)
    n = len(xs)
    if n < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(max(0.0, var))

# ---------------------------
# Context object
# ---------------------------

@dataclass
class SummaryContext:
    """
    Lightweight, tolerant context object shared by summary sections.

    If any of the dirs are omitted (or None), we default to conventional
    relative paths: ./logs, ./models, ./artifacts
    """
    logs_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    artifacts_dir: Optional[Path] = None
    is_demo: bool = False

    # Optional extras used by some sections; kept for backward compatibility
    origins_rows: List[Dict[str, Any]] = field(default_factory=list)
    yield_data: Any = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    caches: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.logs_dir is None:
            self.logs_dir = Path("logs")
        else:
            self.logs_dir = Path(self.logs_dir)

        if self.models_dir is None:
            self.models_dir = Path("models")
        else:
            self.models_dir = Path(self.models_dir)

        if self.artifacts_dir is None:
            self.artifacts_dir = Path("artifacts")
        else:
            self.artifacts_dir = Path(self.artifacts_dir)

        # Ensure dirs exist for sections that write
        ensure_dir(self.logs_dir)
        ensure_dir(self.models_dir)
        ensure_dir(self.artifacts_dir)

# ---------------------------
# Public exports
# ---------------------------

__all__ = [
    "SummaryContext",
    "ensure_dir",
    "_read_text",
    "_write_text",
    "_read_json",
    "_write_json",
    "_load_jsonl",
    "_write_jsonl",
    "parse_ts",
    "_iso",
    "is_demo_mode",
    "generate_demo_yield_plan_if_needed",
    "red",
    "weight_to_label",
    "mean",
    "stdev",
]