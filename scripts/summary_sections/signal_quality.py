# scripts/summary_sections/signal_quality.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, json, bisect
from typing import List, Dict, Tuple

from .common import SummaryContext, is_demo_mode, parse_ts

def _load_jsonl_safe(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
    except Exception:
        return []
    return out

def _bucket_floor(dt: datetime, batch_h: int) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    h = (dt.hour // batch_h) * batch_h
    return dt.replace(hour=h, minute=0, second=0, microsecond=0)

def _iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _classify(precision: float | None) -> Tuple[str, str]:
    # returns (emoji, label)
    if precision is None:
        return "⏳", "Unlabeled"
    if precision >= 0.75:
        return "✅", "Strong"
    if precision >= 0.40:
        return "⚠️", "Mixed"
    return "❌", "Weak"

def _join_labels(
    triggers: List[Tuple[datetime, str]],
    labels_by_origin: Dict[str, List[Tuple[datetime, bool]]],
    window_min: int,
) -> Dict[Tuple[datetime, str], bool | None]:
    """
    For each (ts, origin) trigger, find the nearest label of same origin
    within ±window_min minutes. Returns mapping to label True/False or None if no match.
    Ensures each label row is used at most once.
    """
    out: Dict[Tuple[datetime, str], bool | None] = {}
    used_ix: Dict[str, set[int]] = {o: set() for o in labels_by_origin.keys()}
    wnd = timedelta(minutes=max(0, int(window_min)))
    for ts, origin in triggers:
        arr = labels_by_origin.get(origin, [])
        if not arr:
            out[(ts, origin)] = None
            continue
        # binary search by timestamp
        times = [t for (t, _) in arr]
        i = bisect.bisect_left(times, ts)
        candidates = []
        if i < len(arr): candidates.append(i)
        if i-1 >= 0: candidates.append(i-1)
        # Explore a small neighborhood while within window
        best_ix = None
        best_dt = None
        for ix in sorted(set(candidates + list(range(max(0, i-3), min(len(arr), i+4))))):
            if ix in used_ix[origin]:
                continue
            dti, lab = arr[ix]
            if abs(dti - ts) <= wnd:
                if best_dt is None or abs(dti - ts) < abs(best_dt - ts):
                    best_ix, best_dt = ix, dti
        if best_ix is None:
            out[(ts, origin)] = None
        else:
            used_ix[origin].add(best_ix)
            out[(ts, origin)] = bool(arr[best_ix][1])
    return out

def _persist_json(models_dir: Path, payload: dict):
    try:
        models_dir.mkdir(parents=True, exist_ok=True)
        outp = models_dir / "signal_quality_summary.json"
        outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def _seed_demo(now: datetime, hours: int, batch_h: int) -> List[dict]:
    # 4 demo buckets: Strong, Mixed, Weak, Mixed
    starts = [now - timedelta(hours=h) for h in range(hours, 0, -batch_h)]
    starts = [ _bucket_floor(s, batch_h) for s in starts ][-4:]
    demo = []
    patterns = [
        ("✅","Strong", 8, 7, 1),
        ("⚠️","Mixed",  6, 3, 3),
        ("❌","Weak",   5, 1, 4),
        ("⚠️","Mixed",  7, 4, 3),
    ]
    for i, st in enumerate(starts):
        em, lbl, trig, t, f = patterns[i % len(patterns)]
        prec = t / float(t + f) if (t + f) > 0 else None
        demo.append({
            "start": _iso_utc(st),
            "end": _iso_utc(st + timedelta(hours=batch_h)),
            "triggers": trig,
            "true": t,
            "false": f,
            "precision": prec,
            "class": lbl,
            "emoji": em,
            "demo": True,
        })
    return demo

def append(md: List[str], ctx: SummaryContext, *_, **__):
    """
    Signal Quality Summary (v0.5.5)
    - Look back last 72h (env: MW_SIGNAL_WINDOW_H)
    - Group by batch size (env: MW_SIGNAL_BATCH_H)
    - Join labels within ±5 minutes (env: MW_SIGNAL_JOIN_MIN)
    - Persist models/signal_quality_summary.json
    """
    models_dir = ctx.models_dir
    window_h = int(os.getenv("MW_SIGNAL_WINDOW_H", "72"))
    batch_h  = int(os.getenv("MW_SIGNAL_BATCH_H", "3"))
    join_min = int(os.getenv("MW_SIGNAL_JOIN_MIN", os.getenv("TRAINING_JOIN_WINDOW_MIN", "5")))

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_h)

    trig_path = models_dir / "trigger_history.jsonl"
    lab_path  = models_dir / "label_feedback.jsonl"

    T = _load_jsonl_safe(trig_path)
    L = _load_jsonl_safe(lab_path)

    # Filter & normalize
    triggers: List[Tuple[datetime, str]] = []
    for r in T:
        ts = parse_ts(r.get("timestamp"))
        if not ts or ts < cutoff:
            continue
        origin = str(r.get("origin") or "unknown").lower()
        decision = str(r.get("decision") or "").lower()
        if decision == "triggered":  # only count actual triggers
            triggers.append((ts, origin))

    labels_by_origin: Dict[str, List[Tuple[datetime, bool]]] = {}
    for r in L:
        ts = parse_ts(r.get("timestamp"))
        if not ts or ts < cutoff:
            continue
        origin = str(r.get("origin") or "unknown").lower()
        lab = bool(r.get("label", False))
        labels_by_origin.setdefault(origin, []).append((ts, lab))

    for arr in labels_by_origin.values():
        arr.sort(key=lambda t: t[0])

    # Join labels to triggers
    lab_map = _join_labels(triggers, labels_by_origin, join_min)

    # Aggregate into batches
    batches: Dict[datetime, dict] = {}
    for ts, origin in triggers:
        b = _bucket_floor(ts, batch_h)
        rec = batches.setdefault(b, {"triggers": 0, "true": 0, "false": 0})
        rec["triggers"] += 1
        lab = lab_map.get((ts, origin))
        if lab is True:
            rec["true"] += 1
        elif lab is False:
            rec["false"] += 1
        # None → unlabeled; still counted in triggers

    # Demo seeding if sparse
    seeded = False
    if not batches and is_demo_mode():
        seeded = True
        demo = _seed_demo(now, hours=min(window_h, 24), batch_h=batch_h)
        # Write markdown and persist directly, then return
        md.append(f"\n### 🧪 Signal Quality Summary (last {window_h}h, {batch_h}h buckets)")
        md.append("_(seeded demo data)_")
        for row in demo:
            start = datetime.fromisoformat(row["start"].replace("Z", "+00:00"))
            end   = datetime.fromisoformat(row["end"].replace("Z", "+00:00"))
            rng = f"[{start.strftime('%H:%M')}–{end.strftime('%H:%M')}]"
            em, lbl = row["emoji"], row["class"]
            prec = row["precision"]
            md.append(f"{rng} → {em} {lbl} (precision={prec:.2f}, n={row['triggers']})")
        payload = {
            "window_hours": window_h,
            "batch_hours": batch_h,
            "generated_at": _iso_utc(now),
            "batches": demo,
            "demo": True,
        }
        _persist_json(models_dir, payload)
        return

    # Build ordered output & payload
    ordered_keys = sorted(batches.keys())
    out_rows = []
    for b in ordered_keys:
        stats = batches[b]
        denom = stats["true"] + stats["false"]
        prec = (stats["true"] / float(denom)) if denom > 0 else None
        em, lbl = _classify(prec)
        out_rows.append({
            "start": _iso_utc(b),
            "end": _iso_utc(b + timedelta(hours=batch_h)),
            "triggers": stats["triggers"],
            "true": stats["true"],
            "false": stats["false"],
            "precision": prec,
            "class": lbl,
            "emoji": em,
        })

    md.append(f"\n### 🧪 Signal Quality Summary (last {window_h}h, {batch_h}h buckets)")
    if not out_rows:
        md.append("_No recent triggers/labels._")
    else:
        for row in out_rows:
            start = datetime.fromisoformat(row["start"].replace("Z", "+00:00"))
            end   = datetime.fromisoformat(row["end"].replace("Z", "+00:00"))
            rng = f"[{start.strftime('%H:%M')}–{end.strftime('%H:%M')}]"
            em, lbl, n = row["emoji"], row["class"], row["triggers"]
            prec = row["precision"]
            if prec is None:
                md.append(f"{rng} → {em} {lbl} (no labels, n={n})")
            else:
                md.append(f"{rng} → {em} {lbl} (precision={prec:.2f}, n={n})")

    payload = {
        "window_hours": window_h,
        "batch_hours": batch_h,
        "generated_at": _iso_utc(datetime.now(timezone.utc)),
        "batches": out_rows,
        "demo": False,
    }
    _persist_json(models_dir, payload)