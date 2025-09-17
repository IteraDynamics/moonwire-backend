# scripts/summary_sections/common.py
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, json, hashlib, random, uuid

# ---- Context passed to every section ----
@dataclass
class SummaryContext:
    logs_dir: Path
    models_dir: Path
    is_demo: bool
    origins_rows: list = field(default_factory=list)
    yield_data: dict | None = None
    candidates: list[str] = field(default_factory=list)
    caches: dict = field(default_factory=dict)   # sections may reuse/store computed data


# ---- Generic helpers (moved out of mw_demo_summary) ----
def is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")


def red(s: str) -> str:
    return "000000" if not s else hashlib.sha1(s.encode()).hexdigest()[:6]


def band_weight_from_score(score):
    if score is None:
        return 1.0
    if score >= 0.75:
        return 1.25
    if score >= 0.50:
        return 1.0
    return 0.75


def weight_to_label(w: float) -> str:
    if w >= 1.20:
        return "High"
    if w >= 0.90:
        return "Med"
    return "Low"


def parse_ts(val):
    if val is None:
        return None
    try:
        ts = float(val)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    try:
        s = str(val)
        s = s[:-1] + "+00:00" if s.endswith("Z") else s
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def _iso(dt: datetime) -> str:
    """UTC ISO-8601 with 'Z' and no microseconds, e.g. 2025-09-16T12:34:56Z."""
    return (
        dt.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def pick_candidate_origins(
    origins_rows, yield_data=None, top=3, default=("twitter", "reddit", "rss_news")
):
    seen, out = set(), []
    if yield_data:
        for item in yield_data.get("budget_plan", []) or []:
            o = item.get("origin")
            if o and o != "unknown" and o not in seen:
                out.append(o)
                seen.add(o)
                if len(out) >= top:
                    return out
    for row in origins_rows or []:
        o = row.get("origin")
        if o and o != "unknown" and o not in seen:
            out.append(o)
            seen.add(o)
            if len(out) >= top:
                return out
    for o in default:
        if o not in seen:
            out.append(o)
            seen.add(o)
            if len(out) >= top:
                break
    return out[:top]


# --- DEMO seeders used by a few sections (unchanged behavior) ---
def generate_demo_data_if_needed(reviewers, flag_times=None):
    flag_times = flag_times or []
    if not is_demo_mode() or reviewers:
        return reviewers, []
    now = datetime.now(timezone.utc)
    n = random.randint(3, 5)
    choices = [0.75, 1.0, 1.25]
    seeded, display = [], []
    for _ in range(n):
        rid = f"demo-{uuid.uuid4().hex[:8]}"
        w = random.choice(choices)
        ts = (now - timedelta(minutes=random.randint(2, 55)))
        seeded.append({"id": rid, "weight": w, "timestamp": ts.isoformat()})
        display.append({"id": rid, "weight": w})
        flag_times.append(ts)
    return display, seeded


# Optional explicit export list (helps tests that import specific names)
__all__ = [
    "SummaryContext",
    "is_demo_mode",
    "red",
    "band_weight_from_score",
    "weight_to_label",
    "parse_ts",
    "_iso",
    "pick_candidate_origins",
    "generate_demo_data_if_needed",
]
