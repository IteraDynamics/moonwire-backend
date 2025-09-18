# scripts/summary_sections/common.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Context
# -----------------------------------------------------------------------------
@dataclass
class SummaryContext:
    logs_dir: Path
    models_dir: Path
    is_demo: bool
    origins_rows: List[Dict[str, Any]]  # optional preloaded origin rows
    yield_data: Optional[Dict[str, Any]]  # optional precomputed source yield
    candidates: List[Dict[str, Any]]  # optional preloaded candidates
    caches: Dict[str, Any]            # scratchpad for sections to share


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------
_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

def parse_ts(s: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 string to a UTC-aware datetime; returns None on failure."""
    if not s:
        return None
    try:
        if s.endswith("Z"):
            # try with microseconds first
            try:
                return datetime.strptime(s, _ISO_FMT).replace(tzinfo=timezone.utc)
            except ValueError:
                # fallback without micros
                return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        # generic fallback
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _iso(dt: datetime) -> str:
    """Format a UTC-aware datetime as ISO 8601 with trailing 'Z'."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------------------------
# Filesystem helpers
# -----------------------------------------------------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts; returns [] if missing."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # best-effort; skip bad lines
                continue
    return rows


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------
def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


# -----------------------------------------------------------------------------
# Config / Demo helpers
# -----------------------------------------------------------------------------
def is_demo_mode() -> bool:
    """
    Back-compat helper: read DEMO_MODE from env.
    Many sections now use ctx.is_demo, but some still import this symbol.
    """
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes", "y", "on")


# -----------------------------------------------------------------------------
# Demo seed helpers (used by multiple sections)
# -----------------------------------------------------------------------------
def generate_demo_yield_plan_if_needed(
    ctx: SummaryContext,
    window_hours: int = 168,
) -> Dict[str, Any]:
    """Produce a plausible rate-limit budget plan so the Source Yield section always renders in demo."""
    plan = [
        {"origin": "twitter",  "pct": 0.40},
        {"origin": "reddit",   "pct": 0.35},
        {"origin": "rss_news", "pct": 0.25},
    ]
    return {
        "window_hours": window_hours,
        "plan": plan,
        "demo": True,
    }


def generate_demo_origin_trends_if_needed(
    ctx: SummaryContext,
    window_hours: int = 168,
    interval: str = "hour",  # accept (and mostly ignore) interval to match caller signature
) -> Dict[str, Any]:
    """
    Seed a plausible origin-trend structure so the section always renders in demo.

    Returns:
      {
        "window_hours": int,
        "interval": "hour" | "3h",
        "series": [
          {"origin":"twitter","t": "<iso>", "flags": int, "triggers": int},
          ...
        ],
        "demo": True
      }
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Bound window and step so we never spam too many points
    hours = min(max(int(window_hours), 6), 7 * 24)
    step_h = 1 if interval == "hour" else 3
    points = max(6, min(hours // step_h, 24))

    origins = ["twitter", "reddit", "rss_news"]
    base_flags = {"twitter": 32, "reddit": 20, "rss_news": 14}
    conv = {"twitter": 0.18, "reddit": 0.10, "rss_news": 0.06}

    series: List[Dict[str, Any]] = []
    for o in origins:
        for i in range(points):
            t = now - timedelta(hours=(points - i) * step_h)
            jitter = (i % 3) - 1  # -1,0,1 pattern
            flags = max(0, base_flags.get(o, 10) + jitter)
            triggers = max(0, int(flags * conv.get(o, 0.08)))
            series.append({"origin": o, "t": _iso(t), "flags": flags, "triggers": triggers})

    return {
        "window_hours": hours,
        "interval": interval,
        "series": series,
        "demo": True,
    }