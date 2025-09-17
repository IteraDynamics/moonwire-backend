# scripts/summary_sections/trigger_coverage_summary.py
from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple
from pathlib import Path
from datetime import datetime, timedelta, timezone

from .common import SummaryContext, parse_ts  # removed _iso import

# ----------------------------- small local utils -----------------------------

def _iso(dt: datetime) -> str:
    """UTC ISO-8601 with 'Z' and no microseconds, e.g., 2025-09-16T12:34:56Z."""
    return (
        dt.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

# ----------------------------- helpers -----------------------------

def _load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _scan_candidate_times(logs_dir: Path, t_cut: datetime) -> Dict[str, List[datetime]]:
    """
    Scan logs/ for recent candidate events. Accept any *.jsonl row that has:
      - origin
      - timestamp
      - and a 'candidate-ish' field like 'burst_z' (or count_1h/score) to avoid
        picking up unrelated logs.
    """
    out: Dict[str, List[datetime]] = {}
    if not logs_dir.exists():
        return out

    for p in logs_dir.glob("*.jsonl"):
        for r in _load_jsonl(p):
            ts = parse_ts(r.get("timestamp"))
            if not ts or ts < t_cut:
                continue
            origin = r.get("origin")
            if not origin:
                continue
            if "burst_z" not in r and "count_1h" not in r and "score" not in r:
                continue
            out.setdefault(origin, []).append(ts)

    for k in out:
        out[k].sort()
    return out


def _load_trigger_times(models_dir: Path, t_cut: datetime) -> Dict[str, List[datetime]]:
    """
    Load recent triggers (decision == 'triggered') from models/trigger_history.jsonl
    """
    out: Dict[str, List[datetime]] = {}
    hist = models_dir / "trigger_history.jsonl"
    for r in _load_jsonl(hist):
        ts = parse_ts(r.get("timestamp"))
        if not ts or ts < t_cut:
            continue
        if r.get("decision") != "triggered":
            continue
        origin = r.get("origin")
        if not origin:
            continue
        out.setdefault(origin, []).append(ts)

    for k in out:
        out[k].sort()
    return out


def _count_triggers_matched_to_candidates(
    cand_times: List[datetime],
    trig_times: List[datetime],
    join_min: int,
) -> int:
    """
    Count triggers that have any candidate within ±join_min (greedy two-pointer).
    We count *matched triggers* (upper bound = len(trig_times)).
    """
    if not cand_times or not trig_times:
        return 0
    tol = timedelta(minutes=join_min)
    i = j = 0
    matched = 0
    n_c = len(cand_times)
    n_t = len(trig_times)

    while i < n_c and j < n_t:
        c = cand_times[i]
        t = trig_times[j]
        if abs(c - t) <= tol:
            matched += 1
            i += 1
            j += 1
        elif c < t - tol:
            i += 1
        else:
            j += 1
    return matched


def _classify(rate: float, candidates: int) -> Tuple[str, str]:
    """
    Coverage classes:
      - High (>= 0.15)
      - Medium (>= 0.05)
      - Low (< 0.05)
      - Insufficient if candidates < 3
    """
    if candidates < 3:
        return "Insufficient", "ℹ️"
    if rate >= 0.15:
        return "High", "✅"
    if rate >= 0.05:
        return "Medium", "⚠️"
    return "Low", "❌"


# ----------------------------- public API -----------------------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Compute per-origin trigger coverage over recent window and render markdown.
    Also persists models/trigger_coverage_per_origin.json for downstream use.
    """
    models_dir = ctx.models_dir
    logs_dir = ctx.logs_dir

    window_h = int(os.getenv("MW_TRIGGER_COVERAGE_WINDOW_H", "48"))
    join_min = int(os.getenv("MW_TRIGGER_JOIN_MIN", "5"))

    now = datetime.now(timezone.utc)
    t_cut = now - timedelta(hours=window_h)

    cand_map = _scan_candidate_times(logs_dir, t_cut)
    trig_map = _load_trigger_times(models_dir, t_cut)

    per_origin = []
    origins = sorted(set(cand_map.keys()) | set(trig_map.keys()))
    for origin in origins:
        cand_times = cand_map.get(origin, [])
        trig_times = trig_map.get(origin, [])
        candidates = len(cand_times)
        matched_triggers = _count_triggers_matched_to_candidates(cand_times, trig_times, join_min)
        rate = (matched_triggers / candidates) if candidates > 0 else 0.0
        klass, emoji = _classify(rate, candidates)
        per_origin.append({
            "origin": origin,
            "candidates": candidates,
            "triggers": matched_triggers,
            "trigger_rate": round(rate, 3),
            "class": klass,
            "emoji": emoji,
            "demo": False,
        })

    # Demo seeding if empty and DEMO_MODE is on
    demo_used = False
    if not per_origin and ctx.is_demo:
        per_origin = [
            {"origin":"twitter",  "candidates":55, "triggers":10, "trigger_rate": round(10/55, 3), "class":"High",   "emoji":"✅", "demo":True},
            {"origin":"reddit",   "candidates":42, "triggers": 3, "trigger_rate": round(3/42, 3),  "class":"Medium", "emoji":"⚠️", "demo":True},
            {"origin":"rss_news", "candidates":72, "triggers": 1, "trigger_rate": round(1/72, 3),  "class":"Low",    "emoji":"❌", "demo":True},
        ]
        demo_used = True

    # Sort by class → rate desc → origin
    order = {"High":0, "Medium":1, "Low":2, "Insufficient":3}
    per_origin.sort(key=lambda r: (order.get(r["class"], 9), -r["trigger_rate"], r["origin"]))

    # Persist artifact
    out_json = models_dir / "trigger_coverage_per_origin.json"
    payload = {
        "window_hours": window_h,
        "join_minutes": join_min,
        "generated_at": _iso(now),
        "per_origin": per_origin,
        "demo": demo_used,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # Markdown
    md.append(f"### 📊 Trigger Coverage Summary ({window_h}h){' (demo)' if demo_used else ''}")
    if not per_origin:
        md.append("_no recent coverage data_")
        return

    for r in per_origin:
        pct = f"{r['trigger_rate']*100:.1f}%"
        md.append(f"- `{r['origin']}` → {r['emoji']} {r['class']} (trigger rate = {pct}, {r['triggers']}/{r['candidates']} events)")