# scripts/governance/model_governance_actions.py
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Project helpers
try:
    # canonical imports from the repo
    from scripts.summary_sections.common import SummaryContext, ensure_dir, _iso  # type: ignore
except Exception:  # pragma: no cover - ultra fallback for isolated runs
    @dataclass
    class SummaryContext:  # minimal shim
        logs_dir: Path = Path("logs")
        models_dir: Path = Path("models")
        artifacts_dir: Path = Path("artifacts")
        is_demo: bool = False
        origins_rows: List[Dict[str, Any]] = None  # noqa: ANN401
        yield_data: Any = None
        candidates: List[Dict[str, Any]] = None  # noqa: ANN401
        caches: Dict[str, Any] = None  # noqa: ANN401

    def ensure_dir(p: Path) -> Path:
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _iso(dt: datetime) -> str:
        return dt.replace(microsecond=0, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


# --- compatibility helper for projects where _iso requires dt arg ---
def _iso_now() -> str:
    """Call project _iso() regardless of its signature (with/without dt)."""
    try:
        # Some repos define _iso() -> now
        return _iso()  # type: ignore[arg-type]
    except TypeError:
        # Most MoonWire builds define _iso(dt) -> iso string
        return _iso(datetime.now(timezone.utc))


# --------------------------
# Environment + thresholds
# --------------------------

def _env_bool(name: str, default: bool = False) -> bool:
    return str(os.getenv(name, "true" if default else "false")).strip().lower() in ("1", "true", "yes", "y")

def _mode() -> str:
    v = os.getenv("MW_GOV_ACTION_MODE", "dryrun").strip().lower()
    return "apply" if v == "apply" else "dryrun"

def _thresholds() -> Tuple[float, float]:
    # Targets for decision logic
    try:
        p = float(os.getenv("MW_GOV_ACTION_MIN_PRECISION", "0.75"))
    except Exception:
        p = 0.75
    try:
        e = float(os.getenv("MW_GOV_ACTION_MAX_ECE", "0.06"))
    except Exception:
        e = 0.06
    return p, e


# --------------------------
# IO helpers
# --------------------------

def _read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# --------------------------
# Core decision logic
# --------------------------

def _confidence_from_reasons(base: float, reasons: List[str]) -> float:
    # Simple, deterministic scoring: +0.18 per strong reason
    bumps = 0.0
    for r in reasons:
        if r in ("precision_decline", "precision_improvement", "high_ece", "low_ece"):
            bumps += 0.18
        elif r in ("stable_metrics",):
            bumps += 0.08
    return max(0.0, min(1.0, round(base + bumps, 2)))

def _action_for_version(vinfo: Dict[str, Any], prec_target: float, ece_target: float) -> Dict[str, Any]:
    """
    Decide promote/rollback/observe from trends present in model_performance_trend.json
    Expected keys in vinfo: version, precision_trend, ece_trend, precision_delta, ece_delta
    """
    ver = vinfo.get("version", "unknown")
    p_tr = str(vinfo.get("precision_trend", "stable"))
    e_tr = str(vinfo.get("ece_trend", "stable"))
    p_d  = float(vinfo.get("precision_delta", 0.0) or 0.0)
    e_d  = float(vinfo.get("ece_delta", 0.0) or 0.0)

    # Reason tags (kept simple and test-friendly)
    reasons: List[str] = []
    if p_tr == "declining" or p_d < -0.0001:
        reasons.append("precision_decline")
    if p_tr == "improving" or p_d > 0.0001:
        reasons.append("precision_improvement")
    if e_tr in ("worsening", "declining") or e_d > 0.01:
        reasons.append("high_ece")
    if e_tr in ("improving", "stable") and e_d <= 0:
        reasons.append("low_ece")
    if not reasons:
        reasons.append("stable_metrics")

    # Action rule-of-thumb when absolute ECE not available:
    # - rollback: declining precision & high_ece signal
    # - promote : improving precision & low_ece signal
    # - observe : otherwise
    if ("precision_decline" in reasons) and ("high_ece" in reasons):
        action = "rollback"
        base = 0.5
    elif ("precision_improvement" in reasons) and ("low_ece" in reasons):
        action = "promote"
        base = 0.6
    else:
        action = "observe"
        base = 0.5

    conf = _confidence_from_reasons(base, reasons)

    return {
        "version": ver,
        "action": action,
        "confidence": conf,
        "reason": reasons,
        "demo": False,
    }


# --------------------------
# Demo seeding (ensure ≥1 of each)
# --------------------------

def _demo_seed(actions: List[Dict[str, Any]], versions: List[str]) -> None:
    """Ensure at least one action of each type exists. Deterministic seed."""
    rng_seed = int(os.getenv("MW_DEMO_SEED", "7789"))
    random.seed(rng_seed)

    present = {a["action"] for a in actions}
    need: List[str] = [a for a in ("rollback", "observe", "promote") if a not in present]
    if not need:
        return

    # Create synthetic versions if needed, or reuse existing last version suffixes
    base_versions = versions[:] or ["v0.7.5", "v0.7.6", "v0.7.7"]
    cursor = 8
    for typ in need:
        ver = f"v0.7.{cursor}"
        cursor += 1
        if typ == "rollback":
            synth = {
                "version": ver,
                "precision_trend": "declining",
                "ece_trend": "worsening",
                "precision_delta": round(-0.02 - random.uniform(0.0, 0.02), 3),
                "ece_delta": round(0.012 + random.uniform(0.0, 0.01), 3),
            }
        elif typ == "promote":
            synth = {
                "version": ver,
                "precision_trend": "improving",
                "ece_trend": "improving",
                "precision_delta": round(0.02 + random.uniform(0.0, 0.02), 3),
                "ece_delta": round(-0.01 - random.uniform(0.0, 0.01), 3),
            }
        else:  # observe
            synth = {
                "version": ver,
                "precision_trend": "stable",
                "ece_trend": "stable",
                "precision_delta": 0.0,
                "ece_delta": 0.0,
            }
        a = _action_for_version(synth, * _thresholds())
        actions.append(a)


# --------------------------
# Plotting
# --------------------------

def _write_png_placeholder(path: Path, title_text: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa: F401

        ensure_dir(path.parent)
        fig = plt.figure(figsize=(5, 2.5), dpi=120)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, title_text or "MoonWire", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(str(path))
        plt.close(fig)
        return
    except Exception:
        pass

    # Minimal valid 1x1 PNG bytes if matplotlib not available
    png_1x1 = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
        b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    ensure_dir(path.parent)
    path.write_bytes(png_1x1)

def _plot_actions(actions: List[Dict[str, Any]], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        ensure_dir(out.parent)
        xs = list(range(len(actions)))
        labels = [a["version"] for a in actions]
        colors = {"promote": "#2ca02c", "rollback": "#d62728", "observe": "#7f7f7f"}
        markers = {"promote": "o", "rollback": "X", "observe": "s"}

        fig = plt.figure(figsize=(8, 2.6), dpi=120)
        ax = fig.add_subplot(111)

        for i, a in enumerate(actions):
            ax.scatter(
                i, 0,
                marker=markers.get(a["action"], "o"),
                s=80,
                c=colors.get(a["action"], "#1f77b4"),
                label=a["action"] if i == 0 else None,
                zorder=3,
            )
            ax.text(i, 0.05, a["version"], ha="center", va="bottom", fontsize=8, rotation=0)

        ax.set_yticks([])
        ax.set_xticks(xs)
        ax.set_xticklabels([""] * len(xs))
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_title("Model Governance Actions Timeline")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(str(out))
        plt.close(fig)
    except Exception:
        _write_png_placeholder(out, "model_governance_actions")


# --------------------------
# Public API
# --------------------------

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
        "generated_at": _iso_now(),
        "mode": _mode(),
        "actions": actions,
    }
    (models / "model_governance_actions.json").write_text(json.dumps(plan, indent=2))

    # Plot
    _plot_actions(actions, arts / "model_governance_actions.png")

    # Markdown block
    md.append("🧭 Model Governance Actions (72h)")
    if actions:
        for a in actions:
            ver = a["version"]
            conf = f"{a['confidence']:.2f}"
            reasons = ", ".join(a.get("reason", []))
            md.append(f"{ver} → {a['action']} (confidence {conf}) [{reasons}]")
    else:
        md.append("no actions proposed")
    md.append(f"mode: {_mode()}")