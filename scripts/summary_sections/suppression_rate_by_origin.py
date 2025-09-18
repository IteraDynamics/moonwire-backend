# scripts/summary_sections/suppression_rate_by_origin.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Iterable, Tuple
from bisect import bisect_left
import os, json

from .common import SummaryContext, parse_ts, _iso

# ---- config knobs (env) ----
_DEF_WINDOW_H = 48
_DEF_JOIN_MIN = 5

# ---- simple helpers ----
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0

def _classify(rate: float, candidates: int) -> Tuple[str, str]:
    """
    Buckets:
      ❌ High (≥ 0.80)
      ⚠️ Medium (0.50–0.80)
      ✅ Low (< 0.50)
      ℹ️ Insufficient if candidates < 3
    """
    if candidates < 3:
        return "Insufficient", "ℹ️"
    if rate >= 0.80:
        return "High", "❌"
    if rate >= 0.50:
        return "Medium", "⚠️"
    return "Low", "✅"

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

def _candidate_files(logs_dir: Path) -> List[Path]:
    """
    Heuristic file discovery for candidate event streams.
    Looks for common names used in MoonWire tests/CI.
    """
    names = [
        "candidates.jsonl",
        "candidate_events.jsonl",
        "events.jsonl",
    ]
    explicit = [logs_dir / n for n in names if (logs_dir / n).exists()]
    if explicit:
        return explicit

    # Fallback: any *.jsonl that hints 'cand'
    found = []
    for p in logs_dir.glob("*.jsonl"):
        n = p.name.lower()
        if "candidate" in n or "candidates" in n or n.startswith("cand"):
            found.append(p)
    return found

def _to_epoch_sec(dt: datetime) -> float:
    return dt.timestamp()

def _nearest_match_exists(sorted_epoch_list: List[float], ts_sec: float, tol_sec: float) -> bool:
    """
    Given a sorted list of epoch seconds, check if any entry is within ±tol_sec of ts_sec.
    Uses bisect to find nearest index.
    """
    if not sorted_epoch_list:
        return False
    i = bisect_left(sorted_epoch_list, ts_sec)
    # check left neighbor
    if i > 0 and abs(sorted_epoch_list[i - 1] - ts_sec) <= tol_sec:
        return True
    # check at i
    if i < len(sorted_epoch_list) and abs(sorted_epoch_list[i] - ts_sec) <= tol_sec:
        return True
    return False

def _collect_candidates_by_origin(logs_dir: Path, t_cut: datetime) -> Dict[str, List[float]]:
    """
    Return {origin -> sorted list of candidate timestamps (epoch seconds)} filtered by t_cut.
    """
    out: Dict[str, List[float]] = {}
    for path in _candidate_files(logs_dir):
        for r in _load_jsonl(path):
            ori = (r.get("origin") or "").strip()
            if not ori or ori == "unknown":
                continue
            ts = parse_ts(r.get("timestamp"))
            if not ts:
                continue
            if ts < t_cut:
                continue
            out.setdefault(ori, []).append(_to_epoch_sec(ts))
    # sort each origin list
    for ori, arr in out.items():
        arr.sort()
    return out

def _collect_triggers(models_dir: Path, t_cut: datetime) -> List[Dict[str, Any]]:
    rows = _load_jsonl(models_dir / "trigger_history.jsonl")
    out = []
    for r in rows:
        ori = (r.get("origin") or "").strip()
        if not ori or ori == "unknown":
            continue
        ts = parse_ts(r.get("timestamp"))
        if not ts or ts < t_cut:
            continue
        out.append({"origin": ori, "ts_sec": _to_epoch_sec(ts)})
    return out

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    📉 Suppression Rate by Origin (48h)
      - reddit → ⚠️ Medium (suppression = 68.8%, 11/16)
    Writes models/suppression_rate_per_origin.json
    """
    window_h = int(os.getenv("MW_SUPPRESSION_WINDOW_H", str(_DEF_WINDOW_H)))
    join_min = int(os.getenv("MW_TRIGGER_JOIN_MIN", str(_DEF_JOIN_MIN)))
    tol_sec = float(join_min) * 60.0

    now = datetime.now(timezone.utc)
    t_cut = now - timedelta(hours=window_h)

    # ---- load candidates & triggers
    cand_by_origin = _collect_candidates_by_origin(ctx.logs_dir, t_cut)
    trig_rows = _collect_triggers(ctx.models_dir, t_cut)

    # Count candidates per origin
    candidates_count: Dict[str, int] = {ori: len(ts_list) for ori, ts_list in cand_by_origin.items()}

    # For each trigger, count it only if there exists a candidate within ±join_min
    triggers_count: Dict[str, int] = {}
    for r in trig_rows:
        ori = r["origin"]
        ts_sec = r["ts_sec"]
        if ori not in cand_by_origin:
            # If we never saw candidates for this origin in the window, skip (can't credit a matched trigger)
            continue
        if _nearest_match_exists(cand_by_origin[ori], ts_sec, tol_sec):
            triggers_count[ori] = triggers_count.get(ori, 0) + 1

    # Build rows
    rows: List[Dict[str, Any]] = []
    # Evaluate over the union of origins we've seen candidates for (ignore pure-trigger-only origins; no denominator)
    for ori in sorted(cand_by_origin.keys()):
        c = int(candidates_count.get(ori, 0))
        t = int(triggers_count.get(ori, 0))
        s = max(c - t, 0)
        rate = _safe_div(s, c)
        klass, emoji = _classify(rate, c)
        rows.append({
            "origin": ori,
            "candidates": c,
            "triggers": t,
            "suppressed": s,
            "suppression_rate": round(rate, 3),
            "class": klass,
            "emoji": emoji,
            "demo": False,
        })

    # ---- DEMO fallback: if the window is effectively empty, seed plausible origins
    total_candidates = sum(r["candidates"] for r in rows)
    demo_flag = False
    if total_candidates < 3 and ctx.is_demo:
        demo_flag = True
        rows = [
            # twitter: Medium (≈0.62)
            {
                "origin": "twitter",
                "candidates": 50,
                "triggers": 19,
                "suppressed": 31,
                "suppression_rate": round(31 / 50, 3),
                "class": "Medium",
                "emoji": "⚠️",
                "demo": True,
            },
            # reddit: High (≈0.867)
            {
                "origin": "reddit",
                "candidates": 45,
                "triggers": 6,
                "suppressed": 39,
                "suppression_rate": round(39 / 45, 3),
                "class": "High",
                "emoji": "❌",
                "demo": True,
            },
            # rss_news: Low (≈0.317)
            {
                "origin": "rss_news",
                "candidates": 60,
                "triggers": 41,
                "suppressed": 19,
                "suppression_rate": round(19 / 60, 3),
                "class": "Low",
                "emoji": "✅",
                "demo": True,
            },
        ]

    # Sort by suppression_rate desc so the noisiest suppression is most visible
    rows.sort(key=lambda r: r["suppression_rate"], reverse=True)

    # ---- write artifact
    out_json = ctx.models_dir / "suppression_rate_per_origin.json"
    payload = {
        "window_hours": window_h,
        "join_minutes": join_min,
        "generated_at": _iso(now),
        "per_origin": rows,
        "demo": demo_flag,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- markdown
    md.append(f"### 📉 Suppression Rate by Origin ({window_h}h){' (demo)' if demo_flag else ''}")
    if not rows:
        md.append("_no candidate activity in window_")
        return

    for r in rows:
        pct = round(r["suppression_rate"] * 100.0, 1)
        ori = r["origin"]
        md.append(f"- `{ori}` → {r['emoji']} {r['class']} (suppression = {pct}%, {r['suppressed']}/{r['candidates']})")