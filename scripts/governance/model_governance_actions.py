from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Reuse common helpers if you have them; fall back to light shims if not.
try:
    from scripts.summary_sections.common import SummaryContext, ensure_dir, _iso  # type: ignore
except Exception:
    @dataclass
    class SummaryContext:
        logs_dir: Path | None = None
        models_dir: Path | None = None
        artifacts_dir: Path | None = Path("artifacts")
        is_demo: bool = False

    def ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    from datetime import datetime, timezone
    def _iso(dt: Any = None) -> str:
        dt = dt or datetime.now(timezone.utc)
        return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------- env helpers ----------
def _env_bool(key: str, default: bool) -> bool:
    return str(os.getenv(key, str(default))).lower() in ("1", "true", "yes", "on")

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default

def _mode() -> str:
    m = os.getenv("MW_GOV_ACTION_MODE", "dryrun").lower()
    return "apply" if m == "apply" else "dryrun"


# ---------- io helpers ----------
def _read_json(path: Path) -> Dict[str, Any] | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return None

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2))

# small PNG placeholder (1x1) in case matplotlib is unavailable
_PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
    b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
)

def _write_plot(path: Path, actions: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa
        ensure_dir(path.parent)
        xs = list(range(len(actions)))
        ys = [a.get("confidence", 0.0) for a in actions]
        labels = [f"{a.get('version','?')}:{a.get('action','?')}" for a in actions]
        plt.figure(figsize=(8, 2.5), dpi=120)
        plt.plot(xs, ys, marker="o")
        for x, y, lab in zip(xs, ys, labels):
            plt.text(x, y + 0.03, lab, ha="center", va="bottom", fontsize=8)
        plt.ylim(-0.05, 1.05)
        plt.title("Model Governance Actions — confidence")
        plt.xlabel("action idx")
        plt.ylabel("confidence")
        plt.tight_layout()
        plt.savefig(str(path))
        plt.close()
    except Exception:
        ensure_dir(path.parent)
        path.write_bytes(_PNG_1x1)


# ---------- core logic ----------
def _thresholds() -> Tuple[float, float]:
    # precision target (min acceptable); ECE target (max acceptable)
    prec = _env_float("MW_GOV_ACTION_MIN_PRECISION", 0.75)
    ece = _env_float("MW_GOV_ACTION_MAX_ECE", 0.06)
    return prec, ece


def _confidence(reasons: List[str]) -> float:
    # simple signal count into 0.5..1.0 band
    base = 0.5
    bump = min(len(reasons) * 0.15, 0.5)
    return round(base + bump, 2)


def _action_for_version(vinfo: Dict[str, Any], prec_target: float, ece_target: float) -> Dict[str, Any]:
    ver = vinfo.get("version", "?")
    p_tr = (vinfo.get("precision_trend") or "").lower()
    e_tr = (vinfo.get("ece_trend") or "").lower()
    # deltas optional
    p_d = float(vinfo.get("precision_delta", 0.0) or 0.0)
    e_d = float(vinfo.get("ece_delta", 0.0) or 0.0)

    reasons: List[str] = []
    if p_tr in ("declining", "worsening") or p_d < -0.02:
        reasons.append("precision_decline")
    if e_tr in ("worsening",) or e_d > 0.01:
        reasons.append("high_ece")

    if (p_tr in ("improving",) or p_d > 0.015) and (e_tr in ("improving", "stable") or e_d <= 0.0):
        reasons = ["precision_improvement", "low_ece"]

    # choose action
    if "precision_decline" in reasons and "high_ece" in reasons:
        action = "rollback"
    elif "precision_improvement" in reasons and "low_ece" in reasons:
        action = "promote"
    else:
        action = "observe"
        if not reasons:
            reasons = ["stable_metrics"]

    conf = _confidence(reasons)
    return {
        "version": ver,
        "action": action,
        "confidence": conf,
        "reason": reasons,
        "demo": False,
    }


def _demo_seed(actions: List[Dict[str, Any]], known_versions: List[str]) -> None:
    """
    Ensure we have at least one of each: promote, rollback, observe.
    Non-destructive: only adds if missing.
    Deterministic given process seed.
    """
    want = {"promote", "rollback", "observe"}
    have = {a["action"] for a in actions}
    missing = list(want - have)
    if not missing:
        return

    random.seed(7)  # deterministic
    candidates = known_versions or ["v0.7.5", "v0.7.6", "v0.7.7"]
    for needed in missing:
        ver = random.choice(candidates)
        if needed == "promote":
            reasons = ["precision_improvement", "low_ece"]
            conf = 0.86
        elif needed == "rollback":
            reasons = ["precision_decline", "high_ece"]
            conf = 0.82
        else:
            reasons = ["stable_metrics"]
            conf = 0.5
        actions.append({
            "version": ver,
            "action": needed,
            "confidence": conf,
            "reason": reasons,
            "demo": True,
        })


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build governance actions plan from lineage + trend and append a CI block.
    Writes:
      - models/model_governance_actions.json
      - artifacts/model_governance_actions.png
    """
    models = Path(ctx.models_dir or "models")
    arts = Path(ctx.artifacts_dir or "artifacts")
    ensure_dir(models); ensure_dir(arts)

    # inputs
    lineage = _read_json(models / "model_lineage.json") or {}
    trend = _read_json(models / "model_performance_trend.json") or {}

    versions_from_lineage = [v.get("version") for v in lineage.get("versions", []) if v.get("version")]
    versions_from_trend = [v.get("version") for v in trend.get("versions", []) if v.get("version")]
    all_versions = versions_from_trend or versions_from_lineage

    prec_target, ece_target = _thresholds()
    actions: List[Dict[str, Any]] = []
    trend_map = {v.get("version"): v for v in trend.get("versions", []) if v.get("version")}

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

    # De-dupe by (version, action) keeping highest confidence
    uniq: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for a in actions:
        key = (a["version"], a["action"])
        if key not in uniq or a["confidence"] > uniq[key]["confidence"]:
            uniq[key] = a
    actions = list(uniq.values())

    # JSON plan
    plan = {
        "generated_at": _iso(),
        "mode": _mode(),
        "actions": actions,
    }
    _write_json(models / "model_governance_actions.json", plan)

    # Plot
    _write_plot(arts / "model_governance_actions.png", actions)

    # Markdown block
    md.append("🧭 Model Governance Actions (72h)")
    if actions:
        for a in actions:
            ver = a["version"]; act = a["action"]; conf = a["confidence"]; rsn = ", ".join(a.get("reason", []))
            md.append(f"{ver} → {act} (confidence {conf}) [{rsn}]")
    else:
        md.append("no actions proposed")
    md.append(f"mode: {_mode()}")
