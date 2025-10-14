# scripts/summary_sections/threshold_recommendations.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
import os, json, math, statistics

from .common import SummaryContext, parse_ts, _iso

# ----------------- Config Helpers -----------------
def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)

def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)

def _get_env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on")

# ----------------- IO -----------------
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows

# ----------------- Join -----------------
def _nearest_join_labels_to_triggers(
    labels: List[Dict[str, Any]],
    triggers: List[Dict[str, Any]],
    join_minutes: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any] | None]]:
    """
    For each label, find the nearest trigger on the same origin within ±join_minutes.
    Returns list of (label, trigger_or_none).
    """
    by_origin: Dict[str, List[Dict[str, Any]]] = {}
    for t in triggers:
        o = t.get("origin") or "unknown"
        by_origin.setdefault(o, []).append(t)
    for o in by_origin:
        by_origin[o].sort(key=lambda r: parse_ts(r.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc))

    out: List[Tuple[Dict[str, Any], Dict[str, Any] | None]] = []
    delta = timedelta(minutes=join_minutes)
    for lab in labels:
        o = lab.get("origin") or "unknown"
        tlist = by_origin.get(o, [])
        lts = parse_ts(lab.get("timestamp"))
        best = None
        best_dt = None
        if lts is None:
            out.append((lab, None))
            continue
        # binary search-ish linear scan (lists are small in CI)
        for t in tlist:
            tts = parse_ts(t.get("timestamp"))
            if tts is None:
                continue
            dt = abs(tts - lts)
            if dt <= delta:
                if best_dt is None or dt < best_dt:
                    best, best_dt = t, dt
        out.append((lab, best))
    return out

# ----------------- Metrics / Sweep -----------------
def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0

def _compute_counts_at_threshold(pairs: List[Tuple[float, bool]], thr: float) -> Tuple[int,int,int]:
    """
    pairs: list of (score, label) where label True/False
    Prediction = score >= thr
    Returns (TP, FP, FN). (TN not needed)
    """
    tp = fp = fn = 0
    for s, lab in pairs:
        pred = (s >= thr)
        if lab and pred:
            tp += 1
        elif (not lab) and pred:
            fp += 1
        elif lab and (not pred):
            fn += 1
    return tp, fp, fn

def _sweep_thresholds(
    pairs: List[Tuple[float, bool]],
    strategy: str,
    min_precision: float
) -> Dict[str, Any] | None:
    """
    Returns dict with best {thr, tp, fp, fn, precision, recall, f1, n} or None if no candidates.
    Strategy:
      - "precision_min_recall_max": maximize recall s.t. precision>=min_precision; else max F1
      - "max_f1": maximize F1
    """
    if not pairs:
        return None

    # Candidate thresholds = unique sorted scores (midpoints optional not necessary)
    scores = sorted({float(s) for s, _ in pairs})
    if not scores:
        return None

    # Evaluate each threshold
    cand: List[Dict[str, Any]] = []
    for thr in scores:
        tp, fp, fn = _compute_counts_at_threshold(pairs, thr)
        P = _safe_div(tp, tp + fp)
        R = _safe_div(tp, tp + fn)
        F1 = _safe_div(2*P*R, (P+R)) if (P+R) else 0.0
        cand.append({
            "thr": thr,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": P, "recall": R, "f1": F1,
            "n": tp + fp + fn
        })

    # Filter out degenerate n=0 (no labels)
    cand = [c for c in cand if c["n"] > 0]
    if not cand:
        return None

    if strategy == "precision_min_recall_max":
        feasible = [c for c in cand if c["precision"] >= min_precision]
        if feasible:
            # maximize recall; tie-break by higher F1 then lower|higher thr? Prefer lower thr for recall
            feasible.sort(key=lambda c: (-c["recall"], -c["f1"], c["thr"]))
            return feasible[0]
        # else fall back to max F1
        cand.sort(key=lambda c: (-c["f1"], -c["recall"], -c["precision"], c["thr"]))
        return cand[0]

    # max_f1:
    cand.sort(key=lambda c: (-c["f1"], -c["recall"], -c["precision"], c["thr"]))
    return cand[0]

# ----------------- Classification helpers (for tags/status) -----------------
def _status_tag(delta: float, max_delta: float, allow_large: bool) -> Tuple[str, str]:
    """
    Returns (status, bracket_tag) for markdown.
     - "ok"
     - "within guardrail" (changed less than max_delta)
     - "large-jump (clamped)" (would exceed but clamped)
    """
    ad = abs(delta)
    if ad <= max_delta:
        return "ok", "[within guardrail]" if ad > 0 else "[no change]"
    else:
        return ("ok" if allow_large else "large-jump (clamped)"), ("[large-jump]" if allow_large else "[clamped]")

# ----------------- DEMO seeding -----------------
def _seed_demo_if_needed(per_origin: List[Dict[str, Any]], window_h: int, min_labels: int, min_prec: float) -> List[Dict[str, Any]]:
    if per_origin:
        return per_origin
    # Three plausible origins
    demo = [
        {
            "origin": "twitter",
            "current": 0.50,
            "recommended": 0.47,
            "delta": -0.03,
            "precision": 0.76,
            "recall": 0.71,
            "f1": 0.73,
            "labels": max(min_labels, 12),
            "status": "ok",
            "tag": "[within guardrail]",
            "demo": True,
        },
        {
            "origin": "reddit",
            "current": 0.50,
            "recommended": 0.56,
            "delta": 0.06,
            "precision": 0.78,
            "recall": 0.62,
            "f1": 0.69,
            "labels": max(min_labels, 18),
            "status": "ok",
            "tag": "[within guardrail]",
            "demo": True,
        },
        {
            "origin": "rss_news",
            "current": 0.50,
            "recommended": 0.50,
            "delta": 0.00,
            "precision": 0.75,
            "recall": 0.51,
            "f1": 0.60,
            "labels": max(min_labels, 14),
            "status": "ok",
            "tag": "[no change]",
            "demo": True,
        },
    ]
    return demo

# ----------------- Main -----------------
def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Compute recommended per-origin thresholds from joined label-trigger pairs.
    """
    models_dir = ctx.models_dir
    window_h = _get_env_int("MW_THR_RECO_WINDOW_H", 72)
    join_min = _get_env_int("MW_THRESHOLD_JOIN_MIN", 5)
    min_labels = _get_env_int("MW_THR_RECO_MIN_LABELS", 10)
    min_precision = _get_env_float("MW_THR_RECO_MIN_PREC", 0.75)
    strategy = os.getenv("MW_THR_RECO_STRATEGY", "precision_min_recall_max").strip().lower()
    max_delta = _get_env_float("MW_THR_RECO_MAX_DELTA", 0.10)
    allow_large = _get_env_bool("MW_THR_RECO_ALLOW_LARGE_JUMP", False)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_h)

    # Load rows (cache across sections if available)
    trig_rows = ctx.caches.get("trigger_rows")
    if trig_rows is None:
        trig_rows = _load_jsonl(models_dir / "trigger_history.jsonl")
        ctx.caches["trigger_rows"] = trig_rows
    lab_rows = ctx.caches.get("label_rows")
    if lab_rows is None:
        lab_rows = _load_jsonl(models_dir / "label_feedback.jsonl")
        ctx.caches["label_rows"] = lab_rows

    # Filter by window
    trig_rows = [r for r in trig_rows if (parse_ts(r.get("timestamp")) or now) >= cutoff]
    lab_rows  = [r for r in lab_rows  if (parse_ts(r.get("timestamp")) or now) >= cutoff]

    # Load current thresholds (optional)
    cur_thr: Dict[str, float] = {}
    thr_file = models_dir / "per_origin_thresholds.json"
    if thr_file.exists():
        try:
            cur_thr = json.loads(thr_file.read_text(encoding="utf-8")) or {}
        except Exception:
            cur_thr = {}
    default_thr = 0.50

    # Build joined pairs label↔trigger to read scores by origin
    joined = _nearest_join_labels_to_triggers(lab_rows, trig_rows, join_min)

    # Collect per origin pairs (score, label_bool) and counts
    by_origin_pairs: Dict[str, List[Tuple[float, bool]]] = {}
    by_origin_label_counts: Dict[str, int] = {}

    for lab, trig in joined:
        origin = (lab.get("origin") or (trig or {}).get("origin") or "unknown").strip() or "unknown"
        if origin == "unknown":
            continue
        if trig is None:
            # Matched none → counts toward FN in sweep only if we had score; here we cannot assign a score.
            # We will ignore unmatched labels for sweep (can bias recall upward), but keep count to gate n.
            by_origin_label_counts[origin] = by_origin_label_counts.get(origin, 0) + 1
            continue
        # Use trigger's adjusted_score or score
        score = trig.get("adjusted_score")
        if score is None:
            score = trig.get("score")
        try:
            s = float(score)
        except Exception:
            continue
        lab_val = lab.get("label")
        if lab_val is True:
            by_origin_pairs.setdefault(origin, []).append((s, True))
            by_origin_label_counts[origin] = by_origin_label_counts.get(origin, 0) + 1
        elif lab_val is False:
            by_origin_pairs.setdefault(origin, []).append((s, False))
            by_origin_label_counts[origin] = by_origin_label_counts.get(origin, 0) + 1
        else:
            # skip unlabeled
            pass

    per_origin: List[Dict[str, Any]] = []
    for origin, n_labels in sorted(by_origin_label_counts.items()):
        current = float(cur_thr.get(origin, default_thr))
        pairs = by_origin_pairs.get(origin, [])
        # Only proceed if we have enough labeled pairs with scores
        if len(pairs) < min_labels:
            per_origin.append({
                "origin": origin,
                "current": current,
                "recommended": None,
                "delta": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "labels": int(n_labels),
                "status": "insufficient",
                "tag": "[insufficient]",
                "demo": False,
            })
            continue

        best = _sweep_thresholds(pairs, strategy, min_precision)
        if not best:
            per_origin.append({
                "origin": origin,
                "current": current,
                "recommended": None,
                "delta": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "labels": int(n_labels),
                "status": "insufficient",
                "tag": "[insufficient]",
                "demo": False,
            })
            continue

        rec = float(best["thr"])
        # Guardrails
        raw_delta = rec - current
        status, tag = _status_tag(raw_delta, max_delta, allow_large)
        if not allow_large and abs(raw_delta) > max_delta:
            # clamp recommendation to guardrail edge
            rec = current + (max_delta if raw_delta > 0 else -max_delta)
            raw_delta = rec - current
            status = "large-jump (clamped)"
            tag = "[clamped]"

        per_origin.append({
            "origin": origin,
            "current": round(current, 3),
            "recommended": round(rec, 3),
            "delta": round(raw_delta, 3),
            "precision": round(float(best["precision"]), 2),
            "recall": round(float(best["recall"]), 2),
            "f1": round(float(best["f1"]), 2),
            "labels": int(best["n"]),
            "status": status,
            "tag": tag,
            "demo": False,
        })

    # If totally empty and DEMO on, seed plausible examples
    if not per_origin and ctx.is_demo:
        per_origin = _seed_demo_if_needed(per_origin, window_h=window_h, min_labels=min_labels, min_prec=min_precision)

    # Persist artifact
    out = {
        "window_hours": window_h,
        "join_minutes": join_min,
        "generated_at": _iso(now),
        "objective": {
            "type": ("precision_min_recall_max" if strategy == "precision_min_recall_max" else "max_f1"),
            "min_precision": min_precision if strategy == "precision_min_recall_max" else None,
        },
        "guardrails": {
            "max_delta": max_delta,
            "allow_large_jump": allow_large,
            "min_labels": min_labels,
        },
        "per_origin": per_origin,
        "demo": bool(ctx.is_demo and all(r.get("demo") for r in per_origin)) if per_origin else ctx.is_demo,
    }
    (models_dir).mkdir(parents=True, exist_ok=True)
    (models_dir / "threshold_recommendations.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    # -------- Markdown --------
    hdr_demo = " (demo)" if out.get("demo") else ""
    obj_line = "target P≥{:.2f}; maximize R".format(min_precision) if strategy == "precision_min_recall_max" else "maximize F1"
    md.append(f"### 🎛️ Threshold Recommendations ({window_h}h){hdr_demo}")
    md.append(f"_objective: {obj_line}; guardrail ±{max_delta:.2f}_")
    if not per_origin:
        md.append("_no recommendations (no data)_")
        return

    # Sort: show those with recommended first, then insufficient
    def _sort_key(r: Dict[str, Any]):
        # recommended first, then by abs(delta) desc, then origin
        rec = r.get("recommended")
        return (0 if rec is not None else 1, -abs(r.get("delta") or 0.0), r.get("origin","z"))
    per_origin_sorted = sorted(per_origin, key=_sort_key)

    for r in per_origin_sorted:
        o = r["origin"]
        if r["recommended"] is None:
            md.append(f"- `{o}` → _(insufficient data: n={r['labels']})_")
            continue
        sign = "+" if (r["delta"] or 0) > 0 else ""
        md.append(
            f"- `{o}` → rec={r['recommended']:.2f} (cur={r['current']:.2f}, Δ{sign}{r['delta']:.2f})  "
            f"P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}  {r['tag']}"
        )
