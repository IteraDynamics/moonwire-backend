# scripts/governance/model_governance_actions.py
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Shared utils
from scripts.summary_sections.common import SummaryContext, ensure_dir, _iso

# --- Tiny 1x1 PNG for environments without matplotlib ---
_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
    b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
)

def _safe_plot(points: List[Tuple[int, float]], marks: List[Tuple[int, str]], out: Path) -> None:
    """Try to plot (time,value) with labels; otherwise drop a 1x1 PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa

        ensure_dir(out.parent)
        fig = plt.figure(figsize=(6, 3), dpi=120)
        ax = fig.add_subplot(111)

        if points:
            xs, ys = zip(*points)
            ax.plot(xs, ys, linewidth=1.5)
        for x, label in marks:
            ax.axvline(x, linestyle="--", linewidth=0.8)
            ax.text(x, 0.95, label, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=8)

        ax.set_title("Model Governance Actions")
        ax.set_xlabel("t (index)")
        ax.set_ylabel("signal")
        fig.tight_layout()
        fig.savefig(str(out))
        plt.close(fig)
    except Exception:
        ensure_dir(out.parent)
        out.write_bytes(_PNG_1x1)


def _env_bool(name: str, default: bool) -> bool:
    return str(os.getenv(name, "true" if default else "false")).lower() == "true"


def _get_mode() -> str:
    mode = os.getenv("MW_GOV_ACTION_MODE", "dryrun").strip().lower()
    return "apply" if mode == "apply" else "dryrun"


def _thresholds() -> Tuple[float, float]:
    try:
        p = float(os.getenv("MW_GOV_ACTION_MIN_PRECISION", "0.75"))
    except Exception:
        p = 0.75
    try:
        e = float(os.getenv("MW_GOV_ACTION_MAX_ECE", "0.06"))
    except Exception:
        e = 0.06
    return p, e


def _confidence(signals: List[bool]) -> float:
    if not signals:
        return 0.5
    # simple proportion scaled slightly upward for >2 signals
    base = sum(1 for s in signals if s) / len(signals)
    bonus = 0.05 if len(signals) >= 3 else 0.0
    return max(0.0, min(1.0, round(base + bonus, 2)))


def _action_for_version(v: Dict[str, Any], prec_target: float, ece_target: float) -> Dict[str, Any]:
    """
    Decide governance action using trend labels + deltas.
    We don't assume absolute 'ece' exists; we lean on trend and delta.
    """
    version = v.get("version", "unknown")
    p_trend = v.get("precision_trend", "stable")
    r_trend = v.get("recall_trend", "stable")
    f1_trend = v.get("f1_trend", "stable")
    e_trend = v.get("ece_trend", "stable")

    p_delta = float(v.get("precision_delta", 0.0))
    e_delta = float(v.get("ece_delta", 0.0))

    # Signals
    precision_decline = (p_trend == "declining") or (p_delta < -0.02)
    precision_improve = (p_trend == "improving") or (p_delta > 0.02)
    ece_worsening = (e_trend == "worsening") or (e_delta > 0.01)
    ece_improving = (e_trend == "improving") or (e_delta < -0.005)

    # Core policy
    if precision_decline and ece_worsening:
        action = "rollback"
        reason = []
        if precision_decline: reason.append("precision_decline")
        if ece_worsening: reason.append("high_ece")
        conf = _confidence([precision_decline, ece_worsening])
    elif precision_improve and ece_improving:
        action = "promote"
        reason = []
        if precision_improve: reason.append("precision_improvement")
        if ece_improving: reason.append("low_ece")
        conf = _confidence([precision_improve, ece_improving])
    else:
        action = "observe"
        reason = ["stable_metrics"]
        conf = _confidence([not precision_decline, not ece_worsening])

    return {
        "version": version,
        "action": action,
        "confidence": round(conf, 2),
        "reason": reason,
        "demo": False,
    }


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _demo_seed(actions: List[Dict[str, Any]], versions: List[str]) -> None:
    """
    Ensure at least one of each action type exists in DEMO mode,
    deterministically seeded for repeatability.
    """
    have = {a["action"] for a in actions}
    rnd = random.Random(777)  # deterministic

    # Choose versions to map missing actions; if not enough, invent demo versions.
    pool = list(versions) or ["v_demo_1", "v_demo_2", "v_demo_3"]

    if "rollback" not in have:
        actions.append({
            "version": pool[0] if pool else "v_demo_rollback",
            "action": "rollback",
            "confidence": round(0.8 + 0.1 * rnd.random(), 2),
            "reason": ["precision_decline", "high_ece"],
            "demo": True,
        })
    if "promote" not in have:
        actions.append({
            "version": (pool[1] if len(pool) > 1 else "v_demo_promote"),
            "action": "promote",
            "confidence": round(0.82 + 0.1 * rnd.random(), 2),
            "reason": ["precision_improvement", "low_ece"],
            "demo": True,
        })
    if "observe" not in have:
        actions.append({
            "version": (pool[2] if len(pool) > 2 else "v_demo_observe"),
            "action": "observe",
            "confidence": round(0.55 + 0.1 * rnd.random(), 2),
            "reason": ["stable_metrics"],
            "demo": True,
        })


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build governance actions plan from lineage + trend and append a CI block.
    Writes:
      - models/model_governance_actions.json
      - artifacts/model_governance_actions.png
    """
    models = Path(ctx.models_dir)
    arts = Path(ctx.artifacts_dir)
    ensure_dir(models); ensure_dir(arts)

    # Inputs
    lineage = _read_json(models / "model_lineage.json")
    trend = _read_json(models / "model_performance_trend.json")

    versions_from_lineage = [v.get("version") for v in lineage.get("versions", [])] if lineage else []
    versions_from_trend = [v.get("version") for v in trend.get("versions", [])] if trend else []
    all_versions = [v for v in versions_from_trend if v] or [v for v in versions_from_lineage if v]

    # Build actions
    prec_target, ece_target = _thresholds()
    actions: List[Dict[str, Any]] = []
    trend_map = {v.get("version"): v for v in trend.get("versions", [])} if trend else {}

    if all_versions:
        for ver in all_versions:
            vinfo = trend_map.get(ver, {"version": ver})
            actions.append(_action_for_version(vinfo, prec_target, ece_target))
    else:
        # No inputs present: create a small demo set
        all_versions = ["v0.7.5", "v0.7.6", "v0.7.7"]
        demo_trend = [
            {"version": "v0.7.5", "precision_trend": "declining", "ece_trend": "worsening", "precision_delta": -0.03, "ece_delta": 0.012},
            {"version": "v0.7.6", "precision_trend": "stable", "ece_trend": "stable", "precision_delta": 0.0, "ece_delta": 0.0},
            {"version": "v0.7.7", "precision_trend": "improving", "ece_trend": "improving", "precision_delta": 0.025, "ece_delta": -0.01},
        ]
        for v in demo_trend:
            actions.append(_action_for_version(v, prec_target, ece_target))

    # Demo guarantee: ≥1 of each action type
    if _env_bool("DEMO_MODE", False) or _env_bool("MW_DEMO", False):
        _demo_seed(actions, all_versions)

    # De-dupe by (version, action) while keeping the highest confidence
    uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for a in actions:
        key = (a["version"], a["action"])
        if key not in uniq or a["confidence"] > uniq[key]["confidence"]:
            uniq[key] = a
    actions = list(uniq.values())

    # JSON plan
    plan = {
        "generated_at": _iso(),
        "mode": _get_mode(),
        "actions": actions,
    }
    out_json = models / "model_governance_actions.json"
    out_json.write_text(json.dumps(plan, indent=2))

    # Plot markers along an index line
    idx_points = [(i, 0.5) for i in range(len(actions))]
    idx_marks = []
    for i, a in enumerate(actions):
        tag = {"promote": "PROMOTE", "rollback": "ROLLBACK", "observe": "OBSERVE"}.get(a["action"], a["action"].upper())
        idx_marks.append((i, f"{a['version']} {tag}"))
    _safe_plot(idx_points, idx_marks, arts / "model_governance_actions.png")

    # Markdown block
    md.append("🧭 Model Governance Actions (72h)")
    # Order by version name to keep stable
    def _vkey(a: Dict[str, Any]) -> str: return a.get("version", "")
    for a in sorted(actions, key=_vkey):
        v = a["version"]; act = a["action"]; conf = a["confidence"]
        rsn = ", ".join(a.get("reason", []))
        if rsn:
            md.append(f"{v} → {act} (confidence {conf:.2f}) [{rsn}]")
        else:
            md.append(f"{v} → {act} (confidence {conf:.2f})")
    md.append(f"mode: {_get_mode()}")
