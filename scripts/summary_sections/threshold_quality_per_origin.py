# scripts/summary_sections/threshold_quality_per_origin.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
import json
import math
import os
from typing import Dict, List, Tuple, Any

from .common import SummaryContext, parse_ts

# ---- Config knobs (env) ----
_DEF_WINDOW_H = int(os.getenv("MW_SCORE_WINDOW_H", "48"))
_JOIN_MIN = int(os.getenv("MW_THRESHOLD_JOIN_MIN", "5"))
_MIN_LABELS = int(os.getenv("MW_THRESHOLD_MIN_LABELS", "3"))

# ---- Thresholds file (under models/) ----
_THRESH_FILENAME = "per_origin_thresholds.json"
_OUT_FILENAME = "threshold_quality_per_origin.json"

# ---- Classification buckets ----
def _class_for_f1(f1: float | None, n: int) -> Tuple[str, str]:
    if n < _MIN_LABELS or f1 is None:
        return ("Insufficient", "ℹ️")
    if f1 >= 0.75:
        return ("Strong", "✅")
    if f1 >= 0.40:
        return ("Mixed", "⚠️")
    return ("Weak", "❌")


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


def _score_of(row: dict) -> float | None:
    for k in ("adjusted_score", "score", "prob", "p"):
        v = row.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            return None
    return None


def _load_thresholds(models_dir: Path) -> Dict[str, float]:
    p = models_dir / _THRESH_FILENAME
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return {}


def _closest_trigger(label_ts: datetime, candidates: List[dict]) -> dict | None:
    # candidates are already filtered to same origin and time-window
    best, best_dt = None, None
    for r in candidates:
        ts = parse_ts(r.get("timestamp"))
        if ts is None:
            continue
        dt = abs((ts - label_ts).total_seconds()) / 60.0  # minutes
        if best is None or dt < best_dt:
            best, best_dt = r, dt
    return best


def _join_labels_to_triggers(
    triggers: List[dict],
    labels: List[dict],
    join_minutes: int,
) -> Dict[str, List[Tuple[dict, dict]]]:
    """
    Returns { origin: [(trigger_row, label_row), ...] } where the trigger is the
    closest within ±join_minutes of the label timestamp.
    """
    # Index triggers by origin within a reasonable search window around labels
    by_origin: Dict[str, List[dict]] = {}
    for t in triggers:
        o = t.get("origin") or "unknown"
        by_origin.setdefault(o, []).append(t)

    # sort per origin by timestamp
    for o in by_origin:
        by_origin[o].sort(key=lambda r: parse_ts(r.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))

    pairs: Dict[str, List[Tuple[dict, dict]]] = {}
    for lab in labels:
        o = lab.get("origin") or "unknown"
        lts = parse_ts(lab.get("timestamp"))
        if lts is None:
            continue
        cands = []
        for t in by_origin.get(o, []):
            tts = parse_ts(t.get("timestamp"))
            if tts is None:
                continue
            if abs((tts - lts).total_seconds()) <= join_minutes * 60:
                cands.append(t)
        if not cands:
            continue
        trig = _closest_trigger(lts, cands)
        if trig is None:
            continue
        pairs.setdefault(o, []).append((trig, lab))
    return pairs


def _metrics_from_pairs(pairs: List[Tuple[dict, dict]], threshold: float) -> Tuple[int, int, int]:
    """Return (TP, FP, FN) using trigger score vs threshold and label boolean."""
    tp = fp = fn = 0
    for trig, lab in pairs:
        score = _score_of(trig)
        if score is None:
            # if no score, we can't evaluate; skip
            continue
        pred_pos = score >= threshold
        lab_true = bool(lab.get("label", False))
        if pred_pos and lab_true:
            tp += 1
        elif pred_pos and not lab_true:
            fp += 1
        elif (not pred_pos) and lab_true:
            fn += 1
        else:
            # true negative is not counted in n (by design)
            pass
    return tp, fp, fn


def _safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return a / b


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or p == 0.0 and r == 0.0:
        return None
    denom = (p + r)
    if denom == 0:
        return None
    return 2 * p * r / denom


def _round(x: float | None, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{d}f}"


def _seed_demo() -> Dict[str, Any]:
    # Deterministic-looking demo so CI summaries are stable.
    origins = [
        ("twitter", 0.42, (6, 2, 0)),   # tp, fp, fn  -> Strong
        ("reddit", 0.50, (3, 3, 1)),    # Mixed
        ("rss_news", 0.50, (1, 4, 2)),  # Weak
    ]
    out = {"per_origin": [], "demo": True}
    for o, thr, (tp, fp, fn) in origins:
        p = _safe_div(tp, tp + fp) or 0.0
        r = _safe_div(tp, tp + fn) or 0.0
        f1 = _f1(p, r) or 0.0
        klass, emoji = _class_for_f1(f1, tp + fp + fn)
        out["per_origin"].append({
            "origin": o, "threshold": thr, "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1, "class": klass, "emoji": emoji, "n": tp + fp + fn, "demo": True,
        })
    return out


def append(md: List[str], ctx: SummaryContext):
    """
    Compute per-origin threshold quality metrics over the last window and
    render a compact markdown block + write a JSON artifact.
    """
    now = datetime.now(timezone.utc)
    window_h = int(os.getenv("MW_SCORE_WINDOW_H", str(_DEF_WINDOW_H)))
    join_min = int(os.getenv("MW_THRESHOLD_JOIN_MIN", str(_JOIN_MIN)))
    min_labels = int(os.getenv("MW_THRESHOLD_MIN_LABELS", str(_MIN_LABELS)))
    cutoff = now - timedelta(hours=window_h)

    # ------------- load inputs (cached across sections) -------------
    models_dir = ctx.models_dir
    triggers = ctx.caches.get("trigger_rows")
    if triggers is None:
        triggers = _load_jsonl(models_dir / "trigger_history.jsonl")
        ctx.caches["trigger_rows"] = triggers

    labels = ctx.caches.get("label_rows")
    if labels is None:
        labels = _load_jsonl(models_dir / "label_feedback.jsonl")
        ctx.caches["label_rows"] = labels

    thresholds = _load_thresholds(models_dir)

    # window filter
    def _in_win(row: dict) -> bool:
        ts = parse_ts(row.get("timestamp"))
        return ts is not None and ts >= cutoff and ts <= now

    trig_win = [r for r in triggers if _in_win(r)]
    lab_win = [r for r in labels if _in_win(r)]

    pairs_by_origin = _join_labels_to_triggers(trig_win, lab_win, join_min)

    # ------------- compute per-origin metrics -------------
    per_origin = []
    total_labelled = 0
    for origin, pairs in pairs_by_origin.items():
        thr = float(thresholds.get(origin, 0.5))
        tp, fp, fn = _metrics_from_pairs(pairs, thr)
        n = tp + fp + fn
        total_labelled += n
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = _f1(p, r)
        klass, emoji = _class_for_f1(f1, n)
        per_origin.append({
            "origin": origin,
            "threshold": thr,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p if p is not None else 0.0,
            "recall": r if r is not None else 0.0,
            "f1": f1 if f1 is not None else 0.0,
            "class": klass,
            "emoji": emoji,
            "n": n,
        })

    demo_flag = False
    if total_labelled < min_labels:
        # Seed demo so the section remains visible
        seeded = _seed_demo()
        per_origin = seeded["per_origin"]
        demo_flag = True

    # ------------- sort Strong → Mixed → Weak → Insufficient -------------
    order = {"Strong": 0, "Mixed": 1, "Weak": 2, "Insufficient": 3}
    per_origin.sort(key=lambda d: (order.get(d["class"], 99), d["origin"]))

    # ------------- write artifact -------------
    out = {
        "window_hours": window_h,
        "join_minutes": join_min,
        "min_labels": min_labels,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "per_origin": per_origin,
        "demo": demo_flag,
    }
    (models_dir / _OUT_FILENAME).write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ------------- render markdown -------------
    title = f"### 📊 Per-Origin Threshold Quality ({window_h}h)" + (" (demo)" if demo_flag else "")
    md.append(title)

    if not per_origin:
        md.append("_No labelled data available in window._")
        return

    for row in per_origin:
        o = row["origin"]
        klass, emoji = row["class"], row["emoji"]
        f1 = _round(row["f1"])
        p = _round(row["precision"])
        r = _round(row["recall"])
        n = row["n"]
        thr = _round(row["threshold"], 2)
        md.append(f"- `{o}` → {emoji} {klass} (F1={f1}, P={p}, R={r}, n={n}, threshold={thr})")