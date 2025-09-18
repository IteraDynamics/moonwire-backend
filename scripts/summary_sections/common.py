# scripts/summary_sections/common.py
from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

# -----------------------
# Public dataclass/context
# -----------------------

@dataclass
class SummaryContext:
    logs_dir: Path
    models_dir: Path
    is_demo: bool = False
    origins_rows: List[Dict[str, Any]] = None
    yield_data: Any = None
    candidates: List[Dict[str, Any]] = None
    caches: Dict[str, Any] = None

    def __post_init__(self):
        if self.origins_rows is None:
            self.origins_rows = []
        if self.candidates is None:
            self.candidates = []
        if self.caches is None:
            self.caches = {}


# -----------------------
# Basic time utilities
# -----------------------

def parse_ts(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        # handle Z suffix and fractional seconds
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    # Canonical Z-terminated ISO
    return dt.replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# -----------------------
# Env helpers
# -----------------------

def _truthy(env_val: Optional[str], default: bool = False) -> bool:
    if env_val is None:
        return default
    return str(env_val).strip().lower() in ("1", "true", "yes", "on")


def is_demo_mode() -> bool:
    return _truthy(os.getenv("DEMO_MODE"), default=False)


# -----------------------
# IO helpers
# -----------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
    except Exception:
        pass
    return rows


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_candidates_from_logs(logs_dir: Path) -> List[Dict[str, Any]]:
    """
    Lightweight loader: read any *.jsonl in logs_dir and keep dict rows
    that have at least origin+timestamp. Tests typically write logs/candidates.jsonl.
    """
    out: List[Dict[str, Any]] = []
    if not logs_dir.exists():
        return out
    for p in sorted(logs_dir.glob("*.jsonl")):
        rows = _load_jsonl(p)
        for r in rows:
            if isinstance(r, dict) and "origin" in r and "timestamp" in r:
                out.append(r)
    return out


# -----------------------
# Lightweight color helpers (no-op for CI text)
# -----------------------

def red(s: str) -> str:
    return s

def green(s: str) -> str:
    return s

def yellow(s: str) -> str:
    return s


# -----------------------
# Weight/band helpers
# -----------------------

def band_weight_from_score(score: float) -> str:
    """
    Roughly map a probability-like score to a reviewer weight bucket label.
    (Used only in demo contexts / summaries.)
    """
    if score is None:
        return "Low"
    if score >= 0.8:
        return "High"
    if score >= 0.65:
        return "Med"
    return "Low"


def weight_to_label(weight: str) -> str:
    # Keep identity mapping; extension point if you want fancy labels.
    return (weight or "Med")


# -----------------------
# Demo seed helpers (back-compat)
# -----------------------

def pick_candidate_origins() -> List[str]:
    # Stable 3-origins list used across demo features
    return ["twitter", "reddit", "rss_news"]


def generate_demo_yield_plan_if_needed(**kwargs) -> Dict[str, Any]:
    """
    Very lightweight demo yield plan. Accepts **kwargs to avoid signature mismatches.
    """
    origins = kwargs.get("origins") or pick_candidate_origins()
    rng = random.Random(42)
    weights = {o: round(rng.uniform(0.30, 0.50), 3) for o in origins}
    total = sum(weights.values()) or 1.0
    plan = {o: round(v / total, 3) for o, v in weights.items()}
    return {"plan": plan, "demo": True, "generated_at": _iso(utc_now())}


def generate_demo_origin_trends_if_needed(**kwargs) -> Dict[str, Any]:
    """
    Minimal origin trends demo payload. Accepts **kwargs (interval, window, etc.).
    """
    now = utc_now().replace(minute=0, second=0, microsecond=0)
    origins = kwargs.get("origins") or pick_candidate_origins()
    out = []
    for o in origins:
        out.append({"origin": o, "t": _iso(now - timedelta(days=1)), "flags": 5, "triggers": 2})
        out.append({"origin": o, "t": _iso(now),                   "flags": 7, "triggers": 3})
    return {"series": out, "demo": True, "generated_at": _iso(utc_now())}


# -----------------------
# Demo data seeding (new + legacy compat)
# -----------------------

def _seed_demo_files_with_ctx(
    ctx: SummaryContext,
    window_hours: int = 72,
    join_minutes: int = 5,
) -> Dict[str, Any]:
    """
    New-style seeding that writes minimal files when DEMO is on or ctx.is_demo is True.
    Produces:
      - logs/candidates.jsonl
      - models/trigger_history.jsonl
      - models/label_feedback.jsonl
    Returns a small dict with 'reviewers' and 'events' for header display parity.
    """
    demo = ctx.is_demo or is_demo_mode()
    result = {"reviewers": [], "events": []}
    if not demo:
        return result

    # --- reviewers: 5 stable demo ids/weights
    reviewers = [
        {"id": "96e748", "weight": "Med"},
        {"id": "f066e4", "weight": "Low"},
        {"id": "d09589", "weight": "Low"},
        {"id": "ecf7f6", "weight": "High"},
        {"id": "aecb8d", "weight": "Low"},
    ]

    # --- events: one score per reviewer for header parity
    now = utc_now()
    rng = random.Random(1337)
    events = []
    for r in reviewers:
        base = {"Low": 0.58, "Med": 0.68, "High": 0.82}[r["weight"]]
        jitter = rng.uniform(-0.05, 0.05)
        events.append({
            "signal": r["id"],
            "score": round(max(0.0, min(1.0, base + jitter)), 2),
            "timestamp": _iso(now),
        })

    # --- write minimal logs (only if missing or tiny)
    candidates_path = ctx.logs_dir / "candidates.jsonl"
    triggers_path   = ctx.models_dir / "trigger_history.jsonl"
    feedback_path   = ctx.models_dir / "label_feedback.jsonl"

    ctx.logs_dir.mkdir(parents=True, exist_ok=True)
    ctx.models_dir.mkdir(parents=True, exist_ok=True)

    def _needs_seed(p: Path, min_lines: int = 1) -> bool:
        if not p.exists():
            return True
        try:
            with p.open("r", encoding="utf-8") as f:
                for i, _ in enumerate(f, start=1):
                    if i >= min_lines:
                        return False
        except Exception:
            return True
        return True

    # Seed a tiny set of candidates (3 origins, few timestamps)
    if _needs_seed(candidates_path, min_lines=1):
        now0 = utc_now().replace(second=0, microsecond=0)
        demo_candidates = []
        for o in pick_candidate_origins():
            for m in (5, 10, 15):
                demo_candidates.append({"timestamp": _iso(now0 - timedelta(minutes=m)), "origin": o})
        _append_jsonl(candidates_path, demo_candidates)

    # Seed a couple triggers and labels near the candidate times
    if _needs_seed(triggers_path, min_lines=1):
        now0 = utc_now().replace(second=0, microsecond=0)
        demo_trig = [
            {"timestamp": _iso(now0 - timedelta(minutes=6)), "origin": "twitter", "adjusted_score": 0.72},
            {"timestamp": _iso(now0 - timedelta(minutes=11)), "origin": "reddit",  "adjusted_score": 0.66},
        ]
        _append_jsonl(triggers_path, demo_trig)

    if _needs_seed(feedback_path, min_lines=1):
        now0 = utc_now().replace(second=0, microsecond=0)
        demo_labs = [
            {"timestamp": _iso(now0 - timedelta(minutes=6)), "origin": "twitter", "label": True,  "model_version": "v.test"},
            {"timestamp": _iso(now0 - timedelta(minutes=11)), "origin": "reddit",  "label": False, "model_version": "v.test"},
        ]
        _append_jsonl(feedback_path, demo_labs)

    result["reviewers"] = reviewers
    result["events"] = events
    return result


def generate_demo_data_if_needed(
    ctx_or_reviewers: Union[SummaryContext, List[Dict[str, Any]], Tuple[Any, ...]],
    window_hours: int = 72,
    join_minutes: int = 5,
):
    """
    Back-compat wrapper:
      * If passed a SummaryContext -> returns a dict (and seeds files)  [new API]
      * If passed a list/tuple of reviewers -> returns (reviewers, events) [legacy]
        - DEMO_MODE=false -> ([], [])
        - DEMO_MODE=true  -> (seeded_reviewers, seeded_events) with equal lengths
    """
    # New-style call
    if isinstance(ctx_or_reviewers, SummaryContext):
        return _seed_demo_files_with_ctx(
            ctx_or_reviewers, window_hours=window_hours, join_minutes=join_minutes
        )

    # Legacy call
    reviewers_in = list(ctx_or_reviewers) if isinstance(ctx_or_reviewers, (list, tuple)) else []
    if not is_demo_mode():
        return [], []

    reviewers_out = reviewers_in or [
        {"id": "96e748", "weight": "Med"},
        {"id": "f066e4", "weight": "Low"},
        {"id": "d09589", "weight": "Low"},
        {"id": "ecf7f6", "weight": "High"},
        {"id": "aecb8d", "weight": "Low"},
    ]

    now = utc_now()
    rng = random.Random(202409)
    events: List[Dict[str, Any]] = []
    for i, r in enumerate(reviewers_out):
        rid = r["id"] if isinstance(r, dict) and "id" in r else f"demo_{i:02d}"
        weight = (r.get("weight") if isinstance(r, dict) else "Med") or "Med"
        base = {"Low": 0.58, "Med": 0.68, "High": 0.82}.get(weight, 0.68)
        jitter = rng.uniform(-0.05, 0.05)
        events.append({
            "signal": rid,
            "score": round(max(0.0, min(1.0, base + jitter)), 2),
            "timestamp": _iso(now),
        })

    return reviewers_out, events


# -----------------------
# Module exports
# -----------------------

__all__ = [
    "SummaryContext",
    "parse_ts", "_iso", "utc_now",
    "is_demo_mode",
    "_load_jsonl", "_write_json", "_append_jsonl", "_load_candidates_from_logs",
    "red", "green", "yellow",
    "band_weight_from_score", "weight_to_label",
    "pick_candidate_origins",
    "generate_demo_yield_plan_if_needed",
    "generate_demo_origin_trends_if_needed",
    "generate_demo_data_if_needed",
]