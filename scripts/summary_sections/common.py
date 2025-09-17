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


# --- DEMO seeders used by a few sections ---
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


def generate_demo_yield_plan_if_needed(yield_data: dict | None, origins_rows=None) -> dict | None:
    """
    Provide a synthetic yield plan when running in DEMO_MODE and no real plan exists.
    Structure expected by source_yield_plan.py:
      {
        "budget_plan": [{"origin":"twitter","percent":47.4}, ...],
        "raw_stats": [{"origin":"twitter","flags":10,"triggers":3,"score":0.30}, ...],
        "demo": True
      }
    """
    if not is_demo_mode():
        return yield_data

    if isinstance(yield_data, dict) and isinstance(yield_data.get("budget_plan"), list) and yield_data["budget_plan"]:
        return yield_data

    # Derive candidate origins from recent rows if available; otherwise default.
    origins = []
    if origins_rows:
        seen = set()
        for r in origins_rows:
            o = r.get("origin")
            if o and o not in seen and o != "unknown":
                origins.append(o)
                seen.add(o)
                if len(origins) >= 3:
                    break
    if not origins:
        origins = ["twitter", "reddit", "rss_news"]

    # Random but plausible percentages summing ~100
    weights = [random.uniform(0.2, 1.0) for _ in origins]
    total = sum(weights) or 1.0
    percents = [round(100.0 * w / total, 1) for w in weights]

    budget_plan = [{"origin": o, "percent": p} for o, p in zip(origins, percents)]

    # Raw stats: flags & triggers with a rough score ratio
    raw_stats = []
    for o in origins:
        flags = random.randint(5, 15)
        triggers = max(0, min(flags, int(round(flags * random.uniform(0.05, 0.35)))))
        score = round((triggers / flags) if flags else 0.0, 3)
        raw_stats.append({
            "origin": o,
            "flags": flags,
            "triggers": triggers,
            "score": score,
        })

    return {
        "budget_plan": budget_plan,
        "raw_stats": raw_stats,
        "demo": True,
    }


def generate_demo_origins_if_needed(rows: list | None) -> list:
    """
    Provide a minimal synthetic 'origin breakdown' when in DEMO_MODE and rows are empty.
    Returns a list of dict rows with at least origin + counts that downstream users can read.
    """
    if not is_demo_mode() or (rows and len(rows) > 0):
        return rows or []
    origins = ["twitter", "reddit", "rss_news"]
    out = []
    for o in origins:
        flags = random.randint(1, 5)
        triggers = random.randint(0, flags)
        out.append({"origin": o, "flags": flags, "triggers": triggers})
    return out


def generate_demo_origin_trends_if_needed(trend_rows: list | None, days: int = 7) -> list:
    """
    Provide synthetic per-day trend rows for each origin when in DEMO_MODE and no real rows.
    Expected by origin_trends.py as a flat list of rows:
      {"origin":"reddit","date":"YYYY-MM-DD","flags":N,"triggers":M}
    """
    if not is_demo_mode() or (trend_rows and len(trend_rows) > 0):
        return trend_rows or []

    origins = ["reddit", "rss_news", "twitter"]
    today = datetime.now(timezone.utc).date()
    dates = [(today - timedelta(days=i)).isoformat() for i in range(days, 0, -1)]

    out = []
    for o in origins:
        base = random.randint(2, 8)
        for d in dates:
            # Make it vaguely wavy
            flags = max(0, int(round(base + random.uniform(-2, 4))))
            triggers = max(0, int(round(flags * random.uniform(0.0, 0.6))))
            out.append({"origin": o, "date": d, "flags": flags, "triggers": triggers})
    return out


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
    "generate_demo_yield_plan_if_needed",
    "generate_demo_origins_if_needed",
    "generate_demo_origin_trends_if_needed",
]