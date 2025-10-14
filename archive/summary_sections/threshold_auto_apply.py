# scripts/summary_sections/threshold_auto_apply.py
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .common import SummaryContext, _iso, is_demo_mode


@dataclass
class Guardrails:
    min_precision: float = 0.75
    min_labels: int = 10
    max_delta: float = 0.10
    allow_large_jump: bool = False


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _load_current_thresholds(models_dir: Path) -> Dict[str, float]:
    cur_path = models_dir / "per_origin_thresholds.json"
    data = _read_json(cur_path)
    if isinstance(data, dict):
        # normalize to float
        out = {}
        for k, v in data.items():
            try:
                out[k] = float(v)
            except Exception:
                continue
        return out
    return {}


def _save_current_thresholds(models_dir: Path, thresholds: Dict[str, float]) -> None:
    # Persist as a simple {origin: threshold} file
    _write_json(models_dir / "per_origin_thresholds.json", thresholds)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, str(default).lower())
    return raw.lower() in ("1", "true", "yes", "on")


def _collect_inputs(models_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    reco = _read_json(models_dir / "threshold_recommendations.json") or {}
    btst = _read_json(models_dir / "threshold_backtest.json") or {}
    return reco, btst


def _guardrails_from_env() -> Guardrails:
    return Guardrails(
        min_precision=float(os.getenv("MW_THR_MIN_PRECISION", "0.75")),
        min_labels=int(os.getenv("MW_THR_MIN_LABELS", "10")),
        max_delta=float(os.getenv("MW_THR_MAX_DELTA", "0.10")),
        allow_large_jump=_env_bool("MW_THR_ALLOW_LARGE_JUMP", False),
    )


def _decide_for_origin(
    origin: str,
    cur_thr: float,
    reco_row: Dict[str, Any] | None,
    bt_row: Dict[str, Any] | None,
    guards: Guardrails,
) -> Dict[str, Any]:
    """
    Returns decision record:
      {
        origin, decision: "applied"|"skipped"|"no_change",
        current, recommended, delta,
        precision, recall, f1, labels,
        reason, notes: [...risk flags...]
      }
    """
    # Defaults if we lack structured inputs
    rec_thr = cur_thr
    delta = 0.0
    precision = None
    recall = None
    f1 = None
    labels = None
    notes: List[str] = []

    # Recommended threshold
    if isinstance(reco_row, dict):
        try:
            rec_thr = float(reco_row.get("recommended", rec_thr))
        except Exception:
            rec_thr = rec_thr

    delta = round(rec_thr - cur_thr, 6)

    # Backtest summary for recommended (prefer recommended block, else current/combined)
    if isinstance(bt_row, dict):
        rblk = bt_row.get("recommended")
        if isinstance(rblk, dict):
            precision = rblk.get("precision")
            recall = rblk.get("recall")
            f1 = rblk.get("f1")
            labels = bt_row.get("current", {}).get("triggers")  # often a proxy for joined size
            # Prefer explicit label count if present on bt row
            labels = bt_row.get("labels", labels)
        else:
            # Fall back: take whatever is present
            precision = bt_row.get("precision", precision)
            recall = bt_row.get("recall", recall)
            f1 = bt_row.get("f1", f1)
            labels = bt_row.get("labels", labels)

        # incorporate any explicit risk flags
        risks = bt_row.get("risk") or bt_row.get("risks") or []
        if isinstance(risks, list):
            notes.extend([str(x) for x in risks])

    # If recommendation equals current (within tiny epsilon), mark as no change
    if abs(delta) < 1e-9:
        return {
            "origin": origin,
            "decision": "no_change",
            "current": cur_thr,
            "recommended": rec_thr,
            "delta": 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "labels": int(labels or 0),
            "reason": "no change",
            "notes": notes,
        }

    # Guardrails
    # 1) labels >= min_labels
    if (labels or 0) < guards.min_labels:
        return {
            "origin": origin,
            "decision": "skipped",
            "current": cur_thr,
            "recommended": rec_thr,
            "delta": delta,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "labels": int(labels or 0),
            "reason": "labels<min",
            "notes": notes,
        }

    # 2) precision after change >= target
    if precision is None or float(precision) < guards.min_precision:
        return {
            "origin": origin,
            "decision": "skipped",
            "current": cur_thr,
            "recommended": rec_thr,
            "delta": delta,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "labels": int(labels or 0),
            "reason": "precision<target",
            "notes": notes,
        }

    # 3) delta within guardrail unless allowed
    if abs(delta) > guards.max_delta and not guards.allow_large_jump:
        return {
            "origin": origin,
            "decision": "skipped",
            "current": cur_thr,
            "recommended": rec_thr,
            "delta": delta,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "labels": int(labels or 0),
            "reason": "delta>max_guardrail",
            "notes": notes + ["large-jump"],
        }

    # 4) risk flags from backtest
    lowered = " ".join(notes).lower()
    if any(k in lowered for k in ("precision drop", "risk", "warning")):
        return {
            "origin": origin,
            "decision": "skipped",
            "current": cur_thr,
            "recommended": rec_thr,
            "delta": delta,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "labels": int(labels or 0),
            "reason": "backtest_risk",
            "notes": notes,
        }

    # Passed all guardrails → apply
    return {
        "origin": origin,
        "decision": "applied",
        "current": cur_thr,
        "recommended": rec_thr,
        "delta": delta,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": int(labels or 0),
        "reason": "within_guardrails",
        "notes": notes or ["ok"],
    }


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Reads threshold_recommendations.json + threshold_backtest.json,
    evaluates guardrails, writes:
      - models/per_origin_thresholds.json (updated when applied)
      - models/threshold_auto_apply.json (audit artifact)
    and appends a clear markdown summary.
    """
    models_dir = ctx.models_dir
    demo = ctx.is_demo or is_demo_mode()

    # Load inputs
    reco, btst = _collect_inputs(models_dir)

    # Demo fallback if inputs missing
    if demo and (not isinstance(reco.get("per_origin"), list) or not isinstance(btst.get("per_origin"), list)):
        reco = {
            "window_hours": 72,
            "join_minutes": 5,
            "objective": {"type": "precision_min_recall_max", "min_precision": 0.75},
            "guardrails": {"max_delta": 0.10, "allow_large_jump": False, "min_labels": 10},
            "per_origin": [
                {"origin": "reddit", "current": 0.50, "recommended": 0.56, "delta": 0.06,
                 "precision": 0.78, "recall": 0.62, "f1": 0.69, "labels": 29, "status": "ok", "demo": True},
                {"origin": "twitter","current": 0.50, "recommended": 0.47, "delta": -0.03,
                 "precision": 0.76, "recall": 0.71, "f1": 0.73, "labels": 22, "status": "ok", "demo": True},
                {"origin": "rss_news","current": 0.50, "recommended": 0.50, "delta": 0.00,
                 "precision": 0.75, "recall": 0.51, "f1": 0.60, "labels": 7,  "status": "no_change", "demo": True},
            ],
            "demo": True,
        }
        btst = {
            "window_hours": 72,
            "join_minutes": 5,
            "min_labels": 10,
            "objective": {"type": "precision_min_recall_max", "min_precision": 0.75},
            "per_origin": [
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
                    "current": {"thr": 0.50, "precision": 0.75, "recall": 0.65, "f1": 0.70, "triggers": 10},
                    "recommended": {"thr": 0.47, "precision": 0.76, "recall": 0.71, "f1": 0.73, "triggers": 12},
                    "delta": {"precision": 0.01, "recall": 0.06, "f1": 0.03, "triggers": 2},
                    "risk": ["ok"],
                    "labels": 22,
                    "demo": True,
                },
                {
                    "origin": "rss_news",
                    "current": {"thr": 0.50, "precision": 0.75, "recall": 0.51, "f1": 0.60, "triggers": 8},
                    "recommended": {"thr": 0.50, "precision": 0.75, "recall": 0.51, "f1": 0.60, "triggers": 8},
                    "delta": {"precision": 0.00, "recall": 0.00, "f1": 0.00, "triggers": 0},
                    "risk": ["no change"],
                    "labels": 7,
                    "demo": True,
                },
            ],
            "demo": True,
        }

    reco_rows = {r.get("origin"): r for r in (reco.get("per_origin") or []) if isinstance(r, dict) and r.get("origin")}
    bt_rows = {r.get("origin"): r for r in (btst.get("per_origin") or []) if isinstance(r, dict) and r.get("origin")}

    # Guardrails & context
    guards = _guardrails_from_env()
    window_h = int(reco.get("window_hours") or btst.get("window_hours") or os.getenv("MW_THR_BT_WINDOW_H", "72"))

    # Current thresholds (we may modify these)
    current = _load_current_thresholds(models_dir)

    # Decide per origin present in either recommendations or backtest
    origins = sorted(set(reco_rows.keys()) | set(bt_rows.keys()) | set(current.keys()))

    decisions: List[Dict[str, Any]] = []
    applied_count = 0
    for o in origins:
        cur_thr = float(current.get(o, 0.50))
        decision = _decide_for_origin(
            origin=o,
            cur_thr=cur_thr,
            reco_row=reco_rows.get(o),
            bt_row=bt_rows.get(o),
            guards=guards,
        )
        decisions.append(decision)

        if decision["decision"] == "applied" and not demo:
            # Apply in memory
            current[o] = round(float(decision["recommended"]), 6)
            applied_count += 1

    # Persist updated per_origin_thresholds.json (no-op if nothing changed)
    if not demo:
        _save_current_thresholds(models_dir, current)

    # Build audit artifact with governance fields
    run_id = uuid.uuid4().hex
    audit = {
        "run_id": run_id,
        "applied_by": "auto",
        "applied_at": _iso(datetime.now(timezone.utc)),
        "window_hours": window_h,
        "guardrails": {
            "min_precision": guards.min_precision,
            "min_labels": guards.min_labels,
            "max_delta": guards.max_delta,
            "allow_large_jump": guards.allow_large_jump,
        },
        "per_origin": decisions,
        "summary": {
            "applied": int(sum(1 for d in decisions if d["decision"] == "applied")),
            "skipped": int(sum(1 for d in decisions if d["decision"] == "skipped")),
            "no_change": int(sum(1 for d in decisions if d["decision"] == "no_change")),
        },
        "demo": bool(demo),
    }
    _write_json(models_dir / "threshold_auto_apply.json", audit)

    # -------- markdown output --------
    md.append(f"### 🔒 Threshold Auto-Apply ({window_h}h backtest)")
    md.append(
        f"guardrails: P≥{guards.min_precision:.2f}, "
        f"labels≥{guards.min_labels}, Δ≤±{guards.max_delta:.2f}"
        + (", large jumps allowed" if guards.allow_large_jump else "")
    )

    if not decisions:
        md.append("_no recommendations/backtest available_")
        return

    # Per-origin lines
    for d in decisions:
        o = d["origin"]
        cur = d["current"]
        rec = d["recommended"]
        dp = d.get("precision")
        dr = d.get("recall")
        df1 = d.get("f1")
        reason = d.get("reason", "ok")
        label_n = d.get("labels", 0)
        delta = d.get("delta", 0.0)

        if d["decision"] == "applied":
            md.append(
                f"- `{o}` → **applied** {rec:.2f} (Δ{delta:+.02f}) "
                f"[P={dp if dp is None else f'{float(dp):.2f}'}, R={dr if dr is None else f'{float(dr):.2f}'}, F1={df1 if df1 is None else f'{float(df1):.2f}'}, n={label_n}] "
                f"[{reason}]"
            )
        elif d["decision"] == "no_change":
            md.append(f"- `{o}` → no change ({cur:.2f}) — {reason}")
        else:
            md.append(
                f"- `{o}` → **skipped** {rec:.2f} (current {cur:.2f}, Δ{delta:+.02f}) "
                f"[P={dp if dp is None else f'{float(dp):.2f}'}, R={dr if dr is None else f'{float(dr):.2f}'}, F1={df1 if df1 is None else f'{float(df1):.2f}'}, n={label_n}] "
                f"— {reason}"
            )

    # Footer roll-up
    s = audit["summary"]
    reasons = [d.get("reason", "") for d in decisions if d["decision"] != "applied"]
    # Show top reasons (simple count)
    reason_counts: Dict[str, int] = {}
    for r in reasons:
        if not r:
            continue
        reason_counts[r] = reason_counts.get(r, 0) + 1
    if reason_counts:
        ordered = sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        reason_str = ", ".join(f"{k}×{v}" for k, v in ordered[:4])
    else:
        reason_str = "—"

    md.append(
        f"\n_roll-up_: applied **{s['applied']}**, skipped **{s['skipped']}**, no change **{s['no_change']}** "
        f"(reasons: {reason_str})"
        + (" (demo)" if demo else "")
    )