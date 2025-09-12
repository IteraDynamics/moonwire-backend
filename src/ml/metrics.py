# src/ml/metrics.py
# -*- coding: utf-8 -*-
"""
MoonWire metrics helpers.

Included:
- compute_accuracy_by_version(...): join trigger/label logs and compute
  TP/FP/FN per model_version with precision/recall/F1.
  * Dedup guard: one label counted per trigger (origin + trigger timestamp)
  * Returns per-version stats plus special keys "_micro" and "_macro"
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Iterable, Tuple


# ---------- Internal utilities ----------

def _mw_parse_ts(ts: Any) -> datetime:
    """
    Parse timestamps in:
      - ISO with 'Z' (optionally with fractional seconds)
      - ISO with offset (e.g., '+00:00')
      - epoch seconds (int/float or numeric string)
    Return tz-aware UTC datetime.
    """
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    s = str(ts).strip()
    # numeric string epoch
    if s.replace(".", "", 1).isdigit():
        return datetime.fromtimestamp(float(s), tz=timezone.utc)
    # ISO normalize
    if s.endswith("Z"):
        if "." in s:
            s = s.split(".")[0] + "Z"
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def _iter_jsonl_file(p: Path) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping malformed/blank lines."""
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                # skip malformed
                continue


# ---------- Public API ----------

def compute_accuracy_by_version(
    trigger_log_path: str | Path,
    label_log_path: str | Path,
    window_hours: int = 72,
    match_window_minutes: int = 5,
    dedup_one_label_per_trigger: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Join triggers ↔ feedback by (origin, nearest timestamp within ±match_window)
    and compute TP/FP/FN per model_version, then precision/recall/F1.

    Returns a dict:
      {
        "<version>": {
          "tp": int, "fp": int, "fn": int, "n": int,
          "precision": float, "recall": float, "f1": float
        },
        "_micro": {...},   # overall aggregates across all versions
        "_macro": {...},   # simple mean of per-version metrics
      }

    Notes:
      - TN is omitted by design (requires explicit negatives inventory).
      - When available, the version from the label row (v0.5.2+) is preferred;
        otherwise falls back to the trigger row's model_version; else "unknown".
      - Dedup: if enabled, counts at most one label for a given trigger
        (keyed by (origin, trigger_timestamp)) to prevent double counting.
    """
    trig_p = Path(trigger_log_path)
    lab_p = Path(label_log_path)

    now = datetime.now(timezone.utc)
    t_min = now - timedelta(hours=window_hours)
    win = timedelta(minutes=match_window_minutes)

    # Load & filter within window (best-effort)
    triggers: list[Dict[str, Any]] = []
    for r in _iter_jsonl_file(trig_p) or []:
        ts = r.get("timestamp")
        if not ts:
            continue
        try:
            dt = _mw_parse_ts(ts)
        except Exception:
            continue
        if dt >= t_min:
            r["_dt"] = dt
            triggers.append(r)

    labels: list[Dict[str, Any]] = []
    for r in _iter_jsonl_file(lab_p) or []:
        ts = r.get("timestamp")
        if not ts:
            continue
        try:
            dt = _mw_parse_ts(ts)
        except Exception:
            continue
        if dt >= t_min:
            r["_dt"] = dt
            labels.append(r)

    # Index triggers by origin and sort by time
    by_origin: Dict[str, list[Dict[str, Any]]] = {}
    for r in triggers:
        by_origin.setdefault(r.get("origin", "unknown"), []).append(r)
    for o in by_origin:
        by_origin[o].sort(key=lambda x: x["_dt"])

    # Nearest trigger within ±win (linear scan is OK for typical sizes)
    def _nearest_trigger(origin: str, ldt: datetime) -> Tuple[Dict[str, Any] | None, timedelta | None]:
        rows = by_origin.get(origin, [])
        best, best_abs = None, None
        for tr in rows:
            d = tr["_dt"] - ldt
            ad = abs(d)
            if ad <= win and (best is None or ad < best_abs):
                best, best_abs = tr, ad
        return best, best_abs

    # Tally TP/FP/FN per version
    stats: Dict[str, Dict[str, int]] = {}
    seen_pairs: set[tuple[str, str]] = set()  # (origin, trigger_timestamp)
    for lb in labels:
        origin = lb.get("origin", "unknown")
        ldt = lb["_dt"]
        label_pos = bool(lb.get("label", False))

        tr, _ = _nearest_trigger(origin, ldt)
        if not tr:
            continue  # unmatched label

        trig_ts = tr.get("timestamp")
        if dedup_one_label_per_trigger and isinstance(trig_ts, str):
            key = (origin, trig_ts)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

        decision_pos = bool(tr.get("decision", False))
        version = lb.get("model_version") or tr.get("model_version") or "unknown"
        version = str(version)

        s = stats.setdefault(version, {"tp": 0, "fp": 0, "fn": 0})
        if decision_pos and label_pos:
            s["tp"] += 1
        elif decision_pos and not label_pos:
            s["fp"] += 1
        elif not decision_pos and label_pos:
            s["fn"] += 1
        # TN omitted

    # Per-version metrics
    out: Dict[str, Dict[str, Any]] = {}
    for v, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        n = tp + fp + fn
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out[v] = {
            "tp": tp, "fp": fp, "fn": fn, "n": n,
            "precision": prec, "recall": rec, "f1": f1,
        }

    # Overall aggregates (micro & macro)
    if out:
        # micro across all versions
        gtp = sum(m["tp"] for m in out.values())
        gfp = sum(m["fp"] for m in out.values())
        gfn = sum(m["fn"] for m in out.values())
        gn = gtp + gfp + gfn
        gprec = (gtp / (gtp + gfp)) if (gtp + gfp) > 0 else 0.0
        grec = (gtp / (gtp + gfn)) if (gtp + gfn) > 0 else 0.0
        gf1 = (2 * gprec * grec / (gprec + grec)) if (gprec + grec) > 0 else 0.0
        out["_micro"] = {
            "tp": gtp, "fp": gfp, "fn": gfn, "n": gn,
            "precision": gprec, "recall": grec, "f1": gf1,
        }
        # macro (simple mean of per-version metrics)
        vers = [m for k2, m in out.items() if not str(k2).startswith("_")]
        if vers:
            mprec = sum(m["precision"] for m in vers) / len(vers)
            mrec = sum(m["recall"] for m in vers) / len(vers)
            mf1 = sum(m["f1"] for m in vers) / len(vers)
            out["_macro"] = {
                "precision": mprec, "recall": mrec, "f1": mf1, "versions": len(vers)
            }

    return out
    
    
# Back-compat shim for older callers/tests
def rolling_precision_recall_snapshot(
    trigger_log_path: str | Path,
    label_log_path: str | Path,
    window_hours: int = 72,
    match_window_minutes: int = 5,
    dedup_one_label_per_trigger: bool = True,
) -> Dict[str, Any]:
    """
    Backward-compatible overall (micro) accuracy snapshot across all versions.

    Returns a dict with keys: tp, fp, fn, n, precision, recall, f1
    """
    res = compute_accuracy_by_version(
        trigger_log_path=trigger_log_path,
        label_log_path=label_log_path,
        window_hours=window_hours,
        match_window_minutes=match_window_minutes,
        dedup_one_label_per_trigger=dedup_one_label_per_trigger,
    )
    # Prefer micro aggregate if any matches exist
    micro = res.get("_micro") if isinstance(res, dict) else None
    if micro:
        return {
            "tp": micro["tp"],
            "fp": micro["fp"],
            "fn": micro["fn"],
            "n":  micro["n"],
            "precision": micro["precision"],
            "recall":    micro["recall"],
            "f1":        micro["f1"],
        }
    # No matches -> return zeros
    return {"tp": 0, "fp": 0, "fn": 0, "n": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}