# scripts/summary_sections/common.py
"""
Common helpers shared by summary sections.

Exports
-------
- SummaryContext: lightweight context holder (paths, demo flag, caches)
- ensure_dir(path): create parent dir(s)
- parse_ts(x): parse ISO / epoch seconds / epoch ms / datetime -> aware UTC
- _iso(dt): ISO 8601 Z string
- _load_jsonl(path): list[dict]
- _write_json(path, obj): write JSON (ensures parent dir)
- _write_jsonl(path, rows, append=True): append-or-write JSONL
- is_demo_mode(): True if DEMO_MODE or MW_DEMO are truthy
- weight_to_label(w): "low"|"med"|"high" bucketing
- red/green/yellow(text): minimal emphasis helpers for MD strings
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


# -----------------------------
# Context
# -----------------------------

@dataclass
class SummaryContext:
    logs_dir: Path
    models_dir: Path
    is_demo: bool = False
    # Optional / best-effort extras that some sections may set/use
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    origins_rows: List[Dict[str, Any]] = field(default_factory=list)
    yield_data: Any = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    caches: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Filesystem / JSON helpers
# -----------------------------

def ensure_dir(p: Union[str, Path]) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _write_json(path: Union[str, Path], obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False))


def _load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            # skip malformed
            pass
    return out


def _write_jsonl(path: Union[str, Path], rows: Iterable[Dict[str, Any]], append: bool = True) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Time parsing / formatting
# -----------------------------

def parse_ts(x: Union[str, int, float, datetime]) -> datetime:
    """
    Parse a timestamp into an aware UTC datetime.

    Accepts:
      - ISO-8601 strings (with/without 'Z')
      - epoch seconds (int/float)
      - epoch milliseconds (int/float >= 1e12)
      - datetime (naive -> assumed UTC; aware -> converted to UTC)
    """
    if isinstance(x, datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=timezone.utc)
        return x.astimezone(timezone.utc)

    if isinstance(x, (int, float)):
        xf = float(x)
        # treat big numbers as milliseconds
        if xf >= 1e11:  # ~Sat Mar 03 5138 in seconds; so this is "clearly ms"
            xf = xf / 1000.0
        return datetime.fromtimestamp(xf, tz=timezone.utc)

    if isinstance(x, str):
        s = x.strip()
        # basic Z-normalization
        if s.endswith("Z"):
            try:
                return datetime.fromisoformat(s[:-1]).replace(tzinfo=timezone.utc)
            except Exception:
                pass
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            # maybe it's numeric in a string
            try:
                val = float(s)
                return parse_ts(val)
            except Exception:
                pass
    raise ValueError(f"Unrecognized timestamp: {x!r}")


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# -----------------------------
# Env flags
# -----------------------------

def is_demo_mode() -> bool:
    def _truthy(env: Optional[str]) -> bool:
        return str(env or "").strip().lower() in ("1", "true", "yes", "y", "on")
    return _truthy(os.getenv("DEMO_MODE")) or _truthy(os.getenv("MW_DEMO"))


# -----------------------------
# Presentation helpers
# -----------------------------

def weight_to_label(w: float) -> str:
    """Bucket a 0..1 weight into a human label."""
    try:
        w = float(w)
    except Exception:
        return "low"
    if w < 0.34:
        return "low"
    if w < 0.67:
        return "med"
    return "high"


# Minimal text emphasis helpers. We keep them very simple so they render in CI markdown.
def red(text: str) -> str:
    return f"**{text}**"  # bold as a proxy for emphasis


def green(text: str) -> str:
    return f"**{text}**"


def yellow(text: str) -> str:
    return f"**{text}**"