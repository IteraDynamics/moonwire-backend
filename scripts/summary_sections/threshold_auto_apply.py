# scripts/summary_sections/threshold_auto_apply.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .common import SummaryContext, _iso, is_demo_mode


@dataclass
class Guardrails:
    min_precision: float
    min_labels: int
    max_delta: float
    allow_large_jump: bool


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _round(x: float | None, nd=2) -> float:
    try:
        return round(float(x), nd)
    except Exception:
        return 0.0


def _read_current_thresholds(models_dir: Path, fallback: float = 0.50) -> Dict[str, float]:
    cur_file = models_dir / "per_origin_thresholds.json"
    data = _load_json(cur_file)
    if not isinstance(data, dict):
        return {}
    # normalize to float
    out: Dict[str, float] = {}
    for k, v in data.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            out[str(k)] = fallback
    return out


def _write_current_thresholds(models_dir: Path, thresholds: Dict[str, float]) -> None:
    cur_file = models_dir / "per_origin_thresholds.json"
    # keep keys stable & numeric
    serializable = {k: float(v) for k, v in thresholds.items()}
    _save_json(cur_file, serializable)


def _passes_guardrails(
    cur_thr: float,
    rec_thr: float,
    after_precision: float,
    labels: int,
    risk_flags: List[str],
    g: Guardrails,
) -> Tuple[bool, str]:
    # precision gate
    if after_precision is None or after_precision < g.min_precision:
        return False, f"precision<{g.min_precision:.2f}"
    # label count gate
    if labels < g.min_labels:
        return False, f"labels<{g.min_labels}"
    # large jump gate
    delta = float(rec_thr) - float(cur_thr)
    if not g.allow_large_jump and abs(delta) > g.max_delta:
        return False, "large-jump"
    # risk flags from backtest
    if any(str(r).lower() != "ok" for r in (risk_flags or [])):
        return False, "risk-flag"
    return True, "within guardrails"


def _demo_rows() -> List[Dict[str, Any]]:
    # plausible mix of applied & skipped
    return [
        {
            "origin": "reddit",
            "current": 0.50,
            "recommended": 0.56,
            "delta": 0.06,
            "after": {"precision": 0.78, "recall": 0.62, "f1": 0.69, "labels": 29},
            "risk": ["ok"],
            "decision": "applied",
            "reason": "within guardrails",
            "demo": True,
        },
        {
            "origin": "twitter",
            "current": 0.50,
            "recommended": 0.47,
            "delta": -0.03,
            "after": {"precision": 0.72, "recall": 0.74, "f1": 0.73, "labels": 12},
            "risk": ["precision-drop-risk"],  # force a skip example
            "decision": "skipped",
            "reason": "risk-flag",
            "demo": True,
        },
        {
            "origin": "rss_news",
            "current": 0.50,
            "recommended": 0.50,
            "delta": 0.00,
            "after": {"precision": 0.75, "recall": 0.51, "f1": 0.60, "labels": 18},
            "risk": ["ok"],
            "decision": "skipped",
            "reason": "no-change",
            "demo": True,
        },
    ]


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Auto-apply thresholds per origin with strict guardrails.
    Inputs:
      - models/threshold_recommendations.json  (from v0.6.0)
      - models/threshold_backtest.json         (from v0.6.1)
      - models/per_origin_thresholds.json      (current)
    Output:
      - models/threshold_auto_apply.json (decision log)
      - updates per_origin_thresholds.json if applied
      - CI markdown block
    """
    models_dir = ctx.models_dir

    # Guardrail knobs
    g = Guardrails(
        min_precision=float(os.getenv("MW_THR_MIN_PRECISION", "0.75")),
        min_labels=int(os.getenv("MW_THR_MIN_LABELS", "10")),
        max_delta=float(os.getenv("MW_THR_MAX_DELTA", "0.10")),
        allow_large_jump=os.getenv("MW_THR_ALLOW_LARGE_JUMP", "false").lower() in ("1", "true", "yes"),
    )

    # Load artifacts
    reco = _load_json(models_dir / "threshold_recommendations.json")
    bt   = _load_json(models_dir / "threshold_backtest.json")

    reco_by_origin = {r["origin"]: r for r in (reco.get("per_origin") or []) if isinstance(r, dict) and r.get("origin")}
    bt_by_origin   = {r["origin"]: r for r in (bt.get("per_origin") or []) if isinstance(r, dict) and r.get("origin")}

    window_h = reco.get("window_hours") or bt.get("window_hours") or int(os.getenv("MW_THR_BT_WINDOW_H", "72"))
    objective = reco.get("objective") or bt.get("objective") or {"type": "precision_min_recall_max", "min_precision": g.min_precision}

    # Current thresholds
    cur = _read_current_thresholds(models_dir, fallback=0.50)
    updated = dict(cur)

    decisions: List[Dict[str, Any]] = []

    if not reco_by_origin and not bt_by_origin and (ctx.is_demo or is_demo_mode()):
        # Demo-seed when nothing available
        decisions = _demo_rows()
        # If any "applied", reflect into thresholds in-memory
        for row in decisions:
            if row["decision"] == "applied":
                updated[row["origin"]] = float(row["recommended"])
        demo_flag = True
    else:
        demo_flag = False
        # Evaluate each origin present in recommendations (primary driver)
        all_origins = sorted(set(list(reco_by_origin.keys()) + list(bt_by_origin.keys()) + list(cur.keys())))
        for origin in all_origins:
            rec = reco_by_origin.get(origin)
            bt_row = bt_by_origin.get(origin)
            cur_thr = float(cur.get(origin, 0.50))

            # No recommendation → no-op
            if not rec or "recommended" not in rec:
                decisions.append({
                    "origin": origin,
                    "current": cur_thr,
                    "recommended": cur_thr,
                    "delta": 0.0,
                    "after": {"precision": None, "recall": None, "f1": None, "labels": 0},
                    "risk": ["no-recommendation"],
                    "decision": "skipped",
                    "reason": "no-recommendation",
                    "demo": False,
                })
                continue

            rec_thr = float(rec.get("recommended", cur_thr))
            delta   = _round(rec_thr - cur_thr, 2)

            # Pull "after" metrics from backtest when available
            after_prec = after_rec = after_f1 = None
            labels = 0
            risk_flags: List[str] = []
            if bt_row and isinstance(bt_row.get("recommended"), dict):
                after_prec = bt_row["recommended"].get("precision")
                after_rec  = bt_row["recommended"].get("recall")
                after_f1   = bt_row["recommended"].get("f1")
                labels     = bt_row["recommended"].get("labels") or bt_row.get("current", {}).get("labels") or 0
                # risk comes as list in our v0.6.1 artifact, default to []
                risk_flags = list(bt_row.get("risk") or [])

            # If no backtest info, we can’t safely auto-apply
            if after_prec is None or after_rec is None or after_f1 is None:
                decisions.append({
                    "origin": origin,
                    "current": cur_thr,
                    "recommended": rec_thr,
                    "delta": delta,
                    "after": {"precision": after_prec, "recall": after_rec, "f1": after_f1, "labels": labels},
                    "risk": (risk_flags or []) + ["no-backtest"],
                    "decision": "skipped",
                    "reason": "no-backtest",
                    "demo": False,
                })
                continue

            ok, reason = _passes_guardrails(cur_thr, rec_thr, after_prec, int(labels), risk_flags, g)
            if ok:
                updated[origin] = rec_thr
                decisions.append({
                    "origin": origin,
                    "current": cur_thr,
                    "recommended": rec_thr,
                    "delta": delta,
                    "after": {"precision": _round(after_prec), "recall": _round(after_rec), "f1": _round(after_f1), "labels": int(labels)},
                    "risk": risk_flags or ["ok"],
                    "decision": "applied",
                    "reason": reason,
                    "demo": False,
                })
            else:
                decisions.append({
                    "origin": origin,
                    "current": cur_thr,
                    "recommended": rec_thr,
                    "delta": delta,
                    "after": {"precision": _round(after_prec), "recall": _round(after_rec), "f1": _round(after_f1), "labels": int(labels)},
                    "risk": (risk_flags or []) + [reason],
                    "decision": "skipped",
                    "reason": reason,
                    "demo": False,
                })

    # Persist decisions artifact
    auto_apply_json = {
        "window_hours": int(window_h),
        "generated_at": _iso(datetime.now(timezone.utc)),
        "objective": objective,
        "guardrails": {
            "min_precision": g.min_precision,
            "min_labels": g.min_labels,
            "max_delta": g.max_delta,
            "allow_large_jump": g.allow_large_jump,
        },
        "per_origin": decisions,
        "demo": demo_flag,
    }
    _save_json(models_dir / "threshold_auto_apply.json", auto_apply_json)

    # Write updated thresholds only if we actually applied changes (and not a pure demo run)
    if not demo_flag and any(d.get("decision") == "applied" for d in decisions):
        _write_current_thresholds(models_dir, updated)

    # ---- Markdown ----
    demo_tag = " (demo)" if demo_flag else ""
    md.append(f"### 🔒 Threshold Auto-Apply ({int(window_h)}h backtest){demo_tag}")
    md.append(
        f"guardrails: P≥{g.min_precision:.2f}, labels≥{g.min_labels}, "
        f"Δ≤±{g.max_delta:.2f}{' (large jumps allowed)' if g.allow_large_jump else ''}"
    )

    if not decisions:
        md.append("_no recommendations/backtest available_")
        return

    # table-like lines
    for row in decisions:
        o = row["origin"]
        cur_thr = row["current"]
        rec_thr = row["recommended"]
        dlt = _round(rec_thr - cur_thr, 2)
        af = row.get("after") or {}
        dp = _round(af.get("precision"), 2) if af.get("precision") is not None else None
        dr = _round(af.get("recall"), 2) if af.get("recall") is not None else None
        df1 = _round(af.get("f1"), 2) if af.get("f1") is not None else None
        notes = row.get("reason") or "ok"
        if row.get("decision") == "applied":
            md.append(
                f"- `{o}` → **applied** {rec_thr:.2f} (from {cur_thr:.2f}, Δ{dlt:+.2f}) "
                f"[P={dp if dp is not None else 'n/a'}, R={dr if dr is not None else 'n/a'}, F1={df1 if df1 is not None else 'n/a'}] "
                f"— {notes}"
            )
        elif rec_thr == cur_thr:
            md.append(f"- `{o}` → no change ({cur_thr:.2f}) — {notes}")
        else:
            md.append(
                f"- `{o}` → **skipped** {rec_thr:.2f} (current {cur_thr:.2f}, Δ{dlt:+.2f}) "
                f"[P={dp if dp is not None else 'n/a'}, R={dr if dr is not None else 'n/a'}, F1={df1 if df1 is not None else 'n/a'}] "
                f"— {notes}"
            )
