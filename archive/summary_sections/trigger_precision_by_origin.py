# scripts/summary_sections/trigger_precision_by_origin.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone, timedelta
import os, json, bisect, math

from .common import SummaryContext, parse_ts

# ---------- small local helpers ----------
def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

def _class_from_precision(p: float, n: int) -> Tuple[str, str]:
    if n < 3:
        return "Insufficient", "ℹ️"
    if p >= 0.75:
        return "Strong", "✅"
    if p >= 0.40:
        return "Mixed", "⚠️"
    return "Weak", "❌"

def _nearest_join_labels_to_triggers(
    labels: List[dict],
    triggers: List[dict],
    join_minutes: int
) -> List[Tuple[dict, Optional[dict]]]:
    """
    Greedy nearest join: for each label, find the nearest trigger on SAME origin
    within ±join_minutes. One-to-one: each trigger can match at most one label.
    """
    by_origin_trig: Dict[str, List[Tuple[float, int]]] = {}
    # Prepare per-origin, sorted by timestamp (epoch seconds) with original idx
    for idx, t in enumerate(triggers):
        o = t.get("origin") or "unknown"
        ts = parse_ts(t.get("timestamp"))
        if o == "unknown" or ts is None:
            continue
        by_origin_trig.setdefault(o, []).append((ts.timestamp(), idx))
    for o in by_origin_trig:
        by_origin_trig[o].sort(key=lambda x: x[0])

    used_trigger_idx: set[int] = set()
    out: List[Tuple[dict, Optional[dict]]] = []
    max_delta = join_minutes * 60.0

    for lab in labels:
        o = lab.get("origin") or "unknown"
        if o == "unknown":
            out.append((lab, None))
            continue
        ts = parse_ts(lab.get("timestamp"))
        if ts is None:
            out.append((lab, None))
            continue
        seq = by_origin_trig.get(o) or []
        if not seq:
            out.append((lab, None))
            continue
        # binary search by epoch seconds
        x = ts.timestamp()
        pos = bisect.bisect_left(seq, (x, -1))
        candidates = []
        if pos < len(seq):
            candidates.append(seq[pos])
        if pos > 0:
            candidates.append(seq[pos-1])
        # pick nearest unused within window
        best = None
        best_abs = float("inf")
        for t_epoch, t_idx in candidates:
            if t_idx in used_trigger_idx:
                continue
            d = abs(t_epoch - x)
            if d < best_abs and d <= max_delta:
                best = t_idx
                best_abs = d
        if best is None:
            out.append((lab, None))
        else:
            used_trigger_idx.add(best)
            out.append((lab, triggers[best]))
    return out

# ---------- main section ----------
def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build per-origin trigger precision over the recent window.
    Inputs:
      - models/trigger_history.jsonl
      - models/label_feedback.jsonl
    Output:
      - Markdown section
      - models/trigger_precision_by_origin.json
    """
    models_dir = ctx.models_dir
    window_h = int(os.getenv("MW_TRIGGER_PRECISION_WINDOW_H", "48"))
    join_min = int(os.getenv("MW_TRIGGER_JOIN_MIN", os.getenv("MW_SIGNAL_JOIN_MIN", "5")))

    now = datetime.now(timezone.utc)
    t_cut = now - timedelta(hours=window_h)

    # Reuse caches if present
    trig_rows = ctx.caches.get("trigger_rows")
    lab_rows  = ctx.caches.get("label_rows")

    if trig_rows is None:
        trig_rows = _load_jsonl(models_dir / "trigger_history.jsonl")
        ctx.caches["trigger_rows"] = trig_rows
    if lab_rows is None:
        lab_rows  = _load_jsonl(models_dir / "label_feedback.jsonl")
        ctx.caches["label_rows"] = lab_rows

    # Filter to window & known origin
    def _in_window(r):
        ts = parse_ts(r.get("timestamp"))
        return ts is not None and ts >= t_cut
    def _known_origin(r):
        return (r.get("origin") or "unknown") != "unknown"

    trig_rows_w = [r for r in trig_rows if _in_window(r) and _known_origin(r)]
    lab_rows_w  = [r for r in lab_rows  if _in_window(r) and _known_origin(r)]

    # Join labels -> triggers
    pairs = _nearest_join_labels_to_triggers(lab_rows_w, trig_rows_w, join_min)

    # Per-origin counts (only when a label matched a trigger)
    by_origin: Dict[str, Dict[str, int]] = {}
    for lab, trig in pairs:
        if trig is None:
            continue
        o = (lab.get("origin") or trig.get("origin") or "unknown")
        if o == "unknown":
            continue
        d = by_origin.setdefault(o, {"true":0, "false":0, "n":0})
        if lab.get("label") is True:
            d["true"] += 1
            d["n"] += 1
        elif lab.get("label") is False:
            d["false"] += 1
            d["n"] += 1
        # labels missing/None are ignored

    per_origin: List[Dict[str, Any]] = []
    for o, c in by_origin.items():
        tp = int(c["true"])
        fp = int(c["false"])
        n  = int(c["n"])
        prec = _safe_div(tp, tp + fp)
        klass, emoji = _class_from_precision(prec, n)
        per_origin.append({
            "origin": o,
            "true": tp,
            "false": fp,
            "precision": round(prec, 2),
            "n": n,
            "class": klass,
            "emoji": emoji,
            "demo": False,
        })

    # If empty, seed demo if requested
    demo_used = False
    if not per_origin and ctx.is_demo:
        demo_used = True
        per_origin = [
            {"origin":"twitter",  "true":13, "false":3, "precision":0.81, "n":16, "class":"Strong", "emoji":"✅", "demo":True},
            {"origin":"reddit",   "true":8,  "false":8, "precision":0.50, "n":16, "class":"Mixed",  "emoji":"⚠️", "demo":True},
            {"origin":"rss_news", "true":2,  "false":10,"precision":0.17, "n":12, "class":"Weak",   "emoji":"❌", "demo":True},
        ]

    # Sort by class, then by precision desc, then origin
    order = {"Strong":0, "Mixed":1, "Weak":2, "Insufficient":3}
    per_origin.sort(key=lambda r: (order.get(r["class"], 9), -r["precision"], r["origin"]))

    # Persist artifact
    out_json = models_dir / "trigger_precision_by_origin.json"
    out_json.write_text(json.dumps({
        "window_hours": window_h,
        "join_minutes": join_min,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "per_origin": per_origin,
        "demo": demo_used,
    }, ensure_ascii=False), encoding="utf-8")

    # ---------- markdown ----------
    md.append(f"### 🎯 Trigger Precision by Origin ({window_h}h){' (demo)' if demo_used else ''}")
    if not per_origin:
        md.append("_no labeled triggers in window_")
        return

    for r in per_origin:
        md.append(f"- `{r['origin']}` → {r['emoji']} {r['class']} "
                  f"(precision={r['precision']:.2f}, n={int(r['n'])})")