# scripts/summary_sections/common.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, json, hashlib, random, uuid
from typing import Dict, List, Any, Optional

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

# ---- Generic helpers ----
def is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

def red(s: str) -> str:
    return "000000" if not s else hashlib.sha1(s.encode()).hexdigest()[:6]

def band_weight_from_score(score):
    if score is None: return 1.0
    if score >= 0.75: return 1.25
    if score >= 0.50: return 1.0
    return 0.75

def weight_to_label(w: float) -> str:
    if w >= 1.20: return "High"
    if w >= 0.90: return "Med"
    return "Low"

def parse_ts(val):
    if val is None: return None
    try:
        ts = float(val); return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception: pass
    try:
        s = str(val);  s = s[:-1] + "+00:00" if s.endswith("Z") else s
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception: return None

def pick_candidate_origins(origins_rows, yield_data=None, top=3, default=("twitter","reddit","rss_news")):
    seen, out = set(), []
    if yield_data:
        for item in yield_data.get("budget_plan", []) or []:
            o = item.get("origin")
            if o and o != "unknown" and o not in seen:
                out.append(o); seen.add(o)
                if len(out) >= top: return out
    for row in origins_rows or []:
        o = row.get("origin")
        if o and o != "unknown" and o not in seen:
            out.append(o); seen.add(o)
            if len(out) >= top: return out
    for o in default:
        if o not in seen:
            out.append(o); seen.add(o)
            if len(out) >= top: break
    return out[:top]

# --- DEMO seeders ---
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

def generate_demo_origins_if_needed(origins_rows):
    if not is_demo_mode():
        return origins_rows
    if not origins_rows or all((r.get("origin") == "unknown") for r in origins_rows):
        demo_sources = ["twitter", "reddit", "rss_news"]
        counts = [random.randint(1, 5) for _ in demo_sources]
        total = max(1, sum(counts))
        return [
            {"origin": src, "count": c, "percent": round(c/total*100, 1)}
            for src, c in zip(demo_sources, counts)
        ]
    return origins_rows

def generate_demo_yield_plan_if_needed(yield_data):
    """
    If DEMO_MODE=true and yield_data is empty or has only 'unknown',
    seed a plan so the section is populated.
    """
    if not is_demo_mode():
        return yield_data or {}

    yd = yield_data or {}
    origins = yd.get("origins") or []
    has_budget = bool(yd.get("budget_plan"))
    has_known = any(o.get("origin") != "unknown" for o in origins)

    if has_budget and has_known:
        return yd

    demo_origins = ["twitter", "reddit", "rss_news"]
    demo_flags = [random.randint(5, 15) for _ in demo_origins]
    demo_triggers = [random.randint(1, 4) for _ in demo_origins]
    total_flags = max(1, sum(demo_flags))
    alpha = 0.7

    origins_out = []
    for origin, flags, triggers in zip(demo_origins, demo_flags, demo_triggers):
        trigger_rate = triggers / max(flags, 1)
        volume_share = flags / total_flags
        yield_score = round(alpha * trigger_rate + (1 - alpha) * volume_share, 3)
        origins_out.append({
            "origin": origin,
            "flags": flags,
            "triggers": triggers,
            "trigger_rate": round(trigger_rate, 3),
            "yield_score": yield_score,
            "eligible": True,
        })

    total_yield = sum(o["yield_score"] for o in origins_out) or 1.0
    budget_plan = [
        {"origin": o["origin"], "pct": round(100 * o["yield_score"] / total_yield, 1)}
        for o in sorted(origins_out, key=lambda o: o["yield_score"], reverse=True)
    ]

    return {
        "window_days": 7,
        "totals": {"flags": sum(demo_flags), "triggers": sum(demo_triggers)},
        "origins": origins_out,
        "budget_plan": budget_plan,
        "notes": ["_demo mode: yield plan seeded_"],
    }

def generate_demo_origin_trends_if_needed(trends, days=7, interval="day"):
    if not is_demo_mode():
        return trends or {}
    origins = (trends or {}).get("origins") or []
    if any(o.get("origin") != "unknown" for o in origins):
        return trends
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    def _daily(n):
        out = []
        for i in range(n):
            ts = (now - timedelta(days=(n - 1 - i))).replace(hour=0)
            out.append({
                "timestamp_bucket": ts.isoformat(),
                "flags_count": random.randint(0, 8),
                "triggers_count": random.randint(0, 4),
            })
        return out

    origins_out = []
    for o in ["reddit", "rss_news", "twitter"]:
        origins_out.append({"origin": o, "buckets": _daily(days)})

    return {
        "window_days": days,
        "interval": interval,
        "origins": origins_out,
        "notes": ["demo trends seeded"],
    }