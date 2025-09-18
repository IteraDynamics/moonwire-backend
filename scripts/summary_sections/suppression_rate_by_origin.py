# scripts/summary_sections/suppression_rate_by_origin.py
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Headless plotting not needed here (no chart), so no matplotlib import.
from .common import SummaryContext, parse_ts


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
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
    return rows


def _load_candidates_from_logs(logs_dir: Path) -> List[Dict[str, Any]]:
    """
    Primary source: logs/candidates.jsonl (used by tests).
    Fallback: any *.jsonl in logs/ with 'timestamp' and 'origin'.
    """
    primary = logs_dir / "candidates.jsonl"
    rows = _load_jsonl(primary)
    if rows:
        return rows

    # fallback: union of any jsonl files with timestamp+origin
    all_rows: List[Dict[str, Any]] = []
    if logs_dir.exists():
        for p in logs_dir.glob("*.jsonl"):
            all_rows.extend(_load_jsonl(p))
    # filter to those that look like candidate events
    return [r for r in all_rows if r.get("timestamp") and r.get("origin")]


def _classify(rate: float, candidates: int) -> Tuple[str, str]:
    """
    Buckets:
      ❌ High (≥ 0.80)
      ⚠️ Medium (0.50–0.80)
      ✅ Low (< 0.50)
      ℹ️ Insufficient if candidates < 3
    """
    if candidates < 3:
        return ("Insufficient", "ℹ️")
    if rate >= 0.80:
        return ("High", "❌")
    if rate >= 0.50:
        return ("Medium", "⚠️")
    return ("Low", "✅")


def _pair_counts_within_delta(
    candidate_times: List[datetime],
    trigger_times: List[datetime],
    delta_minutes: int,
) -> int:
    """
    Greedy one-to-one matching of triggers to candidates within ±delta.
    Returns the number of matched triggers (each trigger counted at most once).
    """
    if not candidate_times or not trigger_times:
        return 0
    delta = timedelta(minutes=delta_minutes)

    cts = sorted(candidate_times)
    tts = sorted(trigger_times)

    i = j = 0
    matched = 0
    while i < len(cts) and j < len(tts):
        c, t = cts[i], tts[j]
        if c < t - delta:
            i += 1
        elif c > t + delta:
            j += 1
        else:
            # within window → match them and advance both
            matched += 1
            i += 1
            j += 1
    return matched


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build "Suppression Rate by Origin (48h)" section
    and persist models/suppression_rate_per_origin.json.
    """
    models_dir = ctx.models_dir
    logs_dir = ctx.logs_dir

    window_h = int(os.getenv("MW_SUPPRESSION_WINDOW_H", "48"))
    join_min = int(os.getenv("MW_TRIGGER_JOIN_MIN", "5"))

    now = datetime.now(timezone.utc)
    t_cut = now - timedelta(hours=window_h)

    # Load inputs
    cand_rows = _load_candidates_from_logs(logs_dir)
    trig_rows = _load_jsonl(models_dir / "trigger_history.jsonl")

    # Filter to window and keep minimal fields
    def _norm_rows(rows: Iterable[Dict[str, Any]]) -> List[Tuple[str, datetime]]:
        out: List[Tuple[str, datetime]] = []
        for r in rows:
            origin = r.get("origin") or "unknown"
            ts = parse_ts(r.get("timestamp"))
            if origin and ts and ts >= t_cut:
                out.append((origin, ts))
        return out

    c_pairs = _norm_rows(cand_rows)
    t_pairs = _norm_rows(trig_rows)

    # Group times by origin
    cand_by_o: Dict[str, List[datetime]] = defaultdict(list)
    trig_by_o: Dict[str, List[datetime]] = defaultdict(list)
    for o, ts in c_pairs:
        cand_by_o[o].append(ts)
    for o, ts in t_pairs:
        trig_by_o[o].append(ts)

    # Compute per origin
    per_origin: List[Dict[str, Any]] = []
    all_origins = sorted(set(cand_by_o.keys()) | set(trig_by_o.keys()))
    for origin in all_origins:
        if origin == "unknown":
            continue

        c_times = cand_by_o.get(origin, [])
        t_times = trig_by_o.get(origin, [])

        candidates = len(c_times)
        matched_triggers = _pair_counts_within_delta(c_times, t_times, join_min)
        suppressed = max(0, candidates - matched_triggers)
        rate = suppressed / max(candidates, 1)

        klass, emoji = _classify(rate, candidates)
        per_origin.append(
            {
                "origin": origin,
                "candidates": candidates,
                "triggers": matched_triggers,
                "suppressed": suppressed,
                "suppression_rate": round(rate, 3),
                "class": klass,
                "emoji": emoji,
                "demo": False,
            }
        )

    # If empty and demo enabled, synthesize plausible rows
    demo_used = False
    if not per_origin and ctx.is_demo:
        demo_used = True
        per_origin = [
            {
                "origin": "twitter",
                "candidates": 55,
                "triggers": 10,
                "suppressed": 45,
                "suppression_rate": round(45 / 55, 3),
                "class": "High",
                "emoji": "❌",
                "demo": True,
            },
            {
                "origin": "reddit",
                "candidates": 42,
                "triggers": 3,
                "suppressed": 39,
                "suppression_rate": round(39 / 42, 3),
                "class": "High",
                "emoji": "❌",
                "demo": True,
            },
            {
                "origin": "rss_news",
                "candidates": 72,
                "triggers": 36,
                "suppressed": 36,
                "suppression_rate": round(36 / 72, 3),
                "class": "Medium",
                "emoji": "⚠️",
                "demo": True,
            },
        ]

    # Sort: High → Medium → Low → Insufficient, then by rate desc
    order = {"High": 0, "Medium": 1, "Low": 2, "Insufficient": 3}
    per_origin.sort(key=lambda r: (order.get(r["class"], 9), -r["suppression_rate"], r["origin"]))

    # Persist artifact
    out_json = models_dir / "suppression_rate_per_origin.json"
    out = {
        "window_hours": window_h,
        "join_minutes": join_min,
        "generated_at": _iso(now),
        "per_origin": per_origin,
        "demo": demo_used,
    }
    out_json.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    # Markdown
    md.append(f"### 📉 Suppression Rate by Origin ({window_h}h){' (demo)' if demo_used else ''}")
    if not per_origin:
        md.append("_no suppression data_")
        return

    for r in per_origin:
        pct = round(r["suppression_rate"] * 100, 1)
        md.append(
            f"- `{r['origin']}` → {r['emoji']} {r['class']} "
            f"(suppression = {pct}%, {r['suppressed']}/{r['candidates']})"
        )