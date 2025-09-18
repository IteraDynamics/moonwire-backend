# scripts/summary_sections/common.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import random


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
            try:
                return datetime.strptime(s, _ISO_FMT).replace(tzinfo=timezone.utc)
            except ValueError:
                return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
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
                continue
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], mode: str = "w") -> None:
    ensure_dir(path.parent)
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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
    Prefer ctx.is_demo in new code, but some sections import this symbol.
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


def generate_demo_data_if_needed(
    ctx: SummaryContext,
    window_hours: int = 72,
    join_minutes: int = 5,
) -> Dict[str, Any]:
    """
    Seed minimal-but-plausible demo data files used across sections:
      - logs/candidates.jsonl
      - models/trigger_history.jsonl
      - models/label_feedback.jsonl

    This is intentionally lightweight: only writes when files are missing or tiny.
    """
    demo = ctx.is_demo or is_demo_mode()
    now = datetime.now(timezone.utc)
    t_cut = now - timedelta(hours=window_hours)

    # Ensure dirs
    ensure_dir(ctx.logs_dir)
    ensure_dir(ctx.models_dir)

    # --- Candidates (logs/candidates.jsonl) ---
    cand_path = ctx.logs_dir / "candidates.jsonl"
    cands = _load_jsonl(cand_path)
    if len(cands) < 10 and demo:
        origins = ["twitter", "reddit", "rss_news"]
        seeded: List[Dict[str, Any]] = []
        # ~50–60 rows across last 6 hours
        for o, n in zip(origins, (50, 42, 60)):
            for i in range(n):
                ts = now - timedelta(minutes=random.randint(0, 6 * 60))
                seeded.append({
                    "timestamp": _iso(ts),
                    "origin": o,
                    "burst_z": round(random.uniform(0.0, 4.0), 2),
                })
        # write fresh
        _write_jsonl(cand_path, seeded, mode="w")
        cands = seeded

    # --- Triggers (models/trigger_history.jsonl) ---
    trig_path = ctx.models_dir / "trigger_history.jsonl"
    trigs = _load_jsonl(trig_path)
    if len(trigs) < 6 and demo:
        seeded_t: List[Dict[str, Any]] = []
        # pick a subset of candidates as triggers
        for row in cands[:]:
            if parse_ts(row.get("timestamp")) and row.get("origin") in ("twitter", "reddit", "rss_news"):
                if random.random() < {"twitter": 0.18, "reddit": 0.07, "rss_news": 0.02}.get(row["origin"], 0.05):
                    seeded_t.append({
                        "timestamp": row["timestamp"],
                        "origin": row["origin"],
                        "adjusted_score": round(random.uniform(0.05, 0.95), 2),
                        "decision": "triggered",
                        "model_version": "v.test",
                    })
        if not seeded_t:
            # ensure at least a handful exist
            for o in ("twitter", "reddit", "rss_news"):
                ts = now - timedelta(minutes=random.randint(0, 120))
                seeded_t.append({
                    "timestamp": _iso(ts),
                    "origin": o,
                    "adjusted_score": round(random.uniform(0.3, 0.9), 2),
                    "decision": "triggered",
                    "model_version": "v.test",
                })
        _write_jsonl(trig_path, seeded_t, mode="w")
        trigs = seeded_t

    # --- Label feedback (models/label_feedback.jsonl) ---
    lab_path = ctx.models_dir / "label_feedback.jsonl"
    labs = _load_jsonl(lab_path)
    if len(labs) < 6 and demo:
        seeded_l: List[Dict[str, Any]] = []
        # label ~70% of triggers; 70% true for twitter, 50% reddit, 35% rss
        priors = {"twitter": 0.70, "reddit": 0.50, "rss_news": 0.35}
        for t in trigs:
            ts = parse_ts(t.get("timestamp"))
            if not ts:
                continue
            if random.random() < 0.70:
                o = t.get("origin", "unknown")
                prob_true = priors.get(o, 0.5)
                seeded_l.append({
                    "timestamp": _iso(ts + timedelta(minutes=random.randint(-join_minutes, join_minutes))),
                    "origin": o,
                    "label": bool(random.random() < prob_true),
                    "model_version": t.get("model_version", "v.test"),
                })
        # ensure at least a couple labels exist
        if not seeded_l and trigs:
            t = trigs[0]
            ts = parse_ts(t.get("timestamp")) or now
            seeded_l.append({
                "timestamp": _iso(ts),
                "origin": t.get("origin", "twitter"),
                "label": True,
                "model_version": t.get("model_version", "v.test"),
            })
        _write_jsonl(lab_path, seeded_l, mode="w")
        labs = seeded_l

    return {
        "window_hours": window_hours,
        "join_minutes": join_minutes,
        "generated_at": _iso(now),
        "counts": {
            "candidates": len(cands),
            "triggers": len(trigs),
            "labels": len(labs),
        },
        "paths": {
            "candidates": str(cand_path),
            "trigger_history": str(trig_path),
            "label_feedback": str(lab_path),
        },
        "demo": demo,
    }