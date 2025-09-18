# scripts/summary_sections/threshold_backtest.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timezone, timedelta
import json, os
from collections import defaultdict

# Only rely on helpers that exist in your current common.py
from .common import SummaryContext, parse_ts, _iso, is_demo_mode  # noqa: F401


# ---- local helpers (kept here to avoid depending on extra common helpers) ----
def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
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


def _safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


def _metrics_for_threshold(pairs: List[Tuple[float, bool]], thr: float) -> Dict[str, float]:
    """
    pairs: list of (score, label_bool)
    return: dict with tp, fp, fn, precision, recall, f1, triggers
    """
    tp = fp = fn = 0
    for score, lab in pairs:
        pred_pos = score >= thr
        if lab and pred_pos:
            tp += 1
        elif (not lab) and pred_pos:
            fp += 1
        elif lab and (not pred_pos):
            fn += 1
    P = _safe_div(tp, tp + fp)
    R = _safe_div(tp, tp + fn)
    F1 = _safe_div(2 * P * R, (P + R)) if (P + R) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": P,
        "recall": R,
        "f1": F1,
        "triggers": tp + fp,
    }


def _nearest_join(
    labels: List[Dict[str, Any]],
    triggers: List[Dict[str, Any]],
    join_min: int,
) -> Dict[str, List[Tuple[float, bool]]]:
    """
    Join labels to nearest trigger (same origin) within ± join_min minutes.
    Returns: {origin: [(score, label_bool), ...]}
    """
    by_origin_trig: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
    for t in triggers:
        o = t.get("origin") or "unknown"
        ts = parse_ts(t.get("timestamp"))
        if ts is None:
            continue
        score = t.get("adjusted_score")
        if score is None:
            score = t.get("score")  # fallback, if present
        if score is None:
            continue
        by_origin_trig[o].append((ts, float(score)))
    # sort for binary-like nearest scan later
    for o in by_origin_trig:
        by_origin_trig[o].sort(key=lambda x: x[0])

    out: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
    max_delta = timedelta(minutes=join_min)

    def nearest_score(o: str, ts: datetime) -> Optional[float]:
        arr = by_origin_trig.get(o) or []
        if not arr:
            return None
        # two-pointer style search (linear is fine for small CI windows)
        # Could be optimized, but clarity wins here.
        best = None
        best_dt = None
        for tts, sc in arr:
            dt = abs(tts - ts)
            if best_dt is None or dt < best_dt:
                best, best_dt = sc, dt
        if best_dt is not None and best_dt <= max_delta:
            return best
        return None

    for lb in labels:
        o = lb.get("origin") or "unknown"
        if o == "unknown":
            continue
        ts = parse_ts(lb.get("timestamp"))
        if ts is None:
            continue
        lab_val = lb.get("label")
        if lab_val is None:
            continue
        sc = nearest_score(o, ts)
        if sc is None:
            continue
        out[o].append((float(sc), bool(lab_val)))
    return out


def _class_from_f1(f1: float, n: int) -> Tuple[str, str]:
    # mirror earlier classing for readability, though we mainly use deltas here
    if n < 2:
        return ("Insufficient", "ℹ️")
    if f1 >= 0.75:
        return ("Strong", "✅")
    if f1 >= 0.40:
        return ("Mixed", "⚠️")
    return ("Weak", "❌")


def _format_delta(x: float) -> str:
    # pretty delta with sign and two decimals
    return f"{x:+.02f}"


# ------------------------------------------------------------------------------
def append(md: List[str], ctx: SummaryContext) -> None:
    """
    What-if backtest comparing current per-origin thresholds vs recommended ones
    over a recent window; produces markdown and a JSON artifact.
    """
    models_dir = ctx.models_dir
    logs_dir = ctx.logs_dir  # (not needed here, we join labels↔triggers)
    window_h = int(os.getenv("MW_THR_BT_WINDOW_H", "72"))
    join_min = int(os.getenv("MW_THRESHOLD_JOIN_MIN", "5"))
    min_labels = int(os.getenv("MW_THR_BT_MIN_LABELS", "10"))

    # Objective metadata (best-effort mirrored from v0.6.0)
    # If recommendations file exists, try to extract its objective and guardrails.
    rec_path = models_dir / "threshold_recommendations.json"
    rec_data: Dict[str, Any] = {}
    if rec_path.exists():
        try:
            rec_data = json.loads(rec_path.read_text(encoding="utf-8")) or {}
        except Exception:
            rec_data = {}

    objective = rec_data.get("objective") or {"type": "precision_min_recall_max", "min_precision": 0.75}
    guardrails = rec_data.get("guardrails") or {"max_delta": 0.10, "allow_large_jump": False, "min_labels": 10}

    # Current thresholds
    cur_thr_path = models_dir / "per_origin_thresholds.json"
    try:
        current_thresholds = json.loads(cur_thr_path.read_text(encoding="utf-8")) if cur_thr_path.exists() else {}
    except Exception:
        current_thresholds = {}
    default_thr = 0.50

    # Recommended thresholds
    rec_by_origin: Dict[str, float] = {}
    if isinstance(rec_data.get("per_origin"), list):
        for r in rec_data["per_origin"]:
            o = r.get("origin")
            v = r.get("recommended")
            if o and isinstance(v, (int, float)):
                rec_by_origin[o] = float(v)

    # Load recent labels and triggers and join
    now = datetime.now(timezone.utc)
    tcut = now - timedelta(hours=window_h)
    labels = _load_jsonl(models_dir / "label_feedback.jsonl")
    triggers = _load_jsonl(models_dir / "trigger_history.jsonl")
    labels = [r for r in labels if (parse_ts(r.get("timestamp")) or now) >= tcut]
    triggers = [r for r in triggers if (parse_ts(r.get("timestamp")) or now) >= tcut]

    pairs_by_origin = _nearest_join(labels, triggers, join_min)

    # Compute per-origin current vs recommended metrics
    table_rows: List[str] = []
    out_items: List[Dict[str, Any]] = []

    origins = sorted({*pairs_by_origin.keys(), *current_thresholds.keys(), *rec_by_origin.keys()})
    any_real = False

    for origin in origins:
        if origin == "unknown":
            continue
        pairs = pairs_by_origin.get(origin, [])
        n = len(pairs)
        if n == 0:
            # skip origins with no joined rows unless we need to show “no change” using thresholds;
            # in backtest context it’s clearer to omit
            continue

        any_real = True
        cur_thr = float(current_thresholds.get(origin, default_thr))
        rec_thr = float(rec_by_origin.get(origin, cur_thr))

        cur_metrics = _metrics_for_threshold(pairs, cur_thr)
        rec_metrics = _metrics_for_threshold(pairs, rec_thr)

        dP = rec_metrics["precision"] - cur_metrics["precision"]
        dR = rec_metrics["recall"] - cur_metrics["recall"]
        dF = rec_metrics["f1"] - cur_metrics["f1"]
        dT = rec_metrics["triggers"] - cur_metrics["triggers"]

        # Risk flags
        risks: List[str] = []
        if n < min_labels:
            risks.append("low-n")
        # precision drop while adding triggers
        if (dP < 0 or dF < 0) and dT > 0:
            risks.append("precision_drop")
        # if rec artifact hinted that recommendation was clamped, carry that through
        # Look up rec item for this origin:
        rec_item = None
        for r in (rec_data.get("per_origin") or []):
            if r.get("origin") == origin:
                rec_item = r
                break
        if rec_item:
            status = str(rec_item.get("status", "")).lower()
            if "large" in status or "clamp" in status:
                risks.append("guardrail")

        notes = "ok" if not risks else ", ".join(risks)

        # markdown row
        row = (
            f"{origin:8s} {cur_thr:0.2f} → {rec_thr:0.2f}  "
            f"{_format_delta(dP)} {_format_delta(dR)} {_format_delta(dF)}"
            f" {dT:+d}  {notes}"
        )
        table_rows.append(row)

        out_items.append({
            "origin": origin,
            "current": {
                "thr": cur_thr,
                "precision": round(cur_metrics["precision"], 4),
                "recall": round(cur_metrics["recall"], 4),
                "f1": round(cur_metrics["f1"], 4),
                "triggers": int(cur_metrics["triggers"]),
            },
            "recommended": {
                "thr": rec_thr,
                "precision": round(rec_metrics["precision"], 4),
                "recall": round(rec_metrics["recall"], 4),
                "f1": round(rec_metrics["f1"], 4),
                "triggers": int(rec_metrics["triggers"]),
            },
            "delta": {
                "precision": round(dP, 4),
                "recall": round(dR, 4),
                "f1": round(dF, 4),
                "triggers": int(dT),
            },
            "risk": risks or ["ok"],
            "labels": n,
            "demo": False,
        })

    # DEMO fallback if nothing to show
    demo_used = False
    if not any_real and ctx.is_demo:
        demo_used = True
        # minimal, plausible seeded backtest table
        table_rows = [
            "reddit    0.50 → 0.56  +0.03 +0.08 +0.05  +3  ok",
            "twitter   0.50 → 0.47  +0.01 +0.06 +0.03  +2  ok",
            "rss_news  0.50 → 0.50  +0.00 +0.00 +0.00  +0  no change",
        ]
        out_items = [
            {
                "origin": "reddit",
                "current": {"thr": 0.50, "precision": 0.75, "recall": 0.54, "f1": 0.63, "triggers": 12},
                "recommended": {"thr": 0.56, "precision": 0.78, "recall": 0.62, "f1": 0.69, "triggers": 15},
                "delta": {"precision": 0.03, "recall": 0.08, "f1": 0.05, "triggers": 3},
                "risk": ["ok"],
                "labels": 29,
                "demo": True,
            },
            {
                "origin": "twitter",
                "current": {"thr": 0.50, "precision": 0.74, "recall": 0.65, "f1": 0.69, "triggers": 14},
                "recommended": {"thr": 0.47, "precision": 0.75, "recall": 0.71, "f1": 0.73, "triggers": 16},
                "delta": {"precision": 0.01, "recall": 0.06, "f1": 0.04, "triggers": 2},
                "risk": ["ok"],
                "labels": 22,
                "demo": True,
            },
            {
                "origin": "rss_news",
                "current": {"thr": 0.50, "precision": 0.75, "recall": 0.51, "f1": 0.60, "triggers": 9},
                "recommended": {"thr": 0.50, "precision": 0.75, "recall": 0.51, "f1": 0.60, "triggers": 9},
                "delta": {"precision": 0.00, "recall": 0.00, "f1": 0.00, "triggers": 0},
                "risk": ["no_change"],
                "labels": 18,
                "demo": True,
            },
        ]

    # Persist artifact
    out_json = {
        "window_hours": window_h,
        "join_minutes": join_min,
        "min_labels": min_labels,
        "objective": objective,
        "guardrails": guardrails,
        "per_origin": out_items,
        "generated_at": _iso(datetime.now(timezone.utc)),
        "demo": demo_used,
    }
    (models_dir / "threshold_backtest.json").write_text(json.dumps(out_json, ensure_ascii=False), encoding="utf-8")

    # ---- Markdown block ----
    hdr_demo = " (demo)" if demo_used else ""
    md.append(f"### 🧪 Threshold Backtest ({window_h}h){hdr_demo}")
    # Objective line
    if objective.get("type") == "precision_min_recall_max":
        md.append(f"objective: P≥{objective.get('min_precision', 0.75):.2f}, maximize R")
    elif objective.get("type") == "max_f1":
        md.append("objective: maximize F1")

    if not table_rows:
        md.append("_no backtest data_")
        return

    # Pretty-ish fixed-width table in plaintext
    md.append("")
    md.append("origin   cur thr → rec   ΔP     ΔR     ΔF1    Δtriggers  notes")
    for row in table_rows:
        md.append(row)
