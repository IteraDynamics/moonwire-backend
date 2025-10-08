# scripts/governance/model_governance_actions.py
from __future__ import annotations
import json, os, random
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List
from scripts.summary_sections.common import ensure_dir, _iso, SummaryContext

def _now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def _trend_label(v: float) -> str:
    if v < -0.02:
        return "declining"
    if v > 0.02:
        return "improving"
    return "stable"

def _seed_demo_actions() -> List[Dict[str, Any]]:
    """Create deterministic demo data with promote/rollback/observe."""
    random.seed(77)
    return [
        {
            "version": "v0.7.5",
            "action": "rollback",
            "confidence": 0.84,
            "reason": ["precision_decline", "high_ece"],
            "demo": True,
        },
        {
            "version": "v0.7.6",
            "action": "observe",
            "confidence": 0.55,
            "reason": ["stable_metrics"],
            "demo": True,
        },
        {
            "version": "v0.7.7",
            "action": "promote",
            "confidence": 0.91,
            "reason": ["precision_improvement", "low_ece"],
            "demo": True,
        },
    ]

def _decide_action(ptrend: str, ece: float, min_p: float, max_ece: float) -> tuple[str, List[str], float]:
    """Governance logic for single version."""
    reasons = []
    conf = random.uniform(0.5, 0.95)

    if ptrend == "declining" and ece > max_ece:
        reasons = ["precision_decline", "high_ece"]
        return "rollback", reasons, round(conf, 2)
    if ptrend == "improving" and ece < max_ece:
        reasons = ["precision_improvement", "low_ece"]
        return "promote", reasons, round(conf, 2)
    return "observe", ["stable_metrics"], round(conf * 0.8, 2)

def append(md: List[str], ctx: SummaryContext) -> None:
    models = Path(ctx.models_dir)
    arts = Path(ctx.artifacts_dir)
    ensure_dir(models); ensure_dir(arts)

    lineage = _read_json(models / "model_lineage.json")
    trend = _read_json(models / "model_performance_trend.json")

    versions = []
    if trend and trend.get("versions"):
        versions = [v.get("version") for v in trend["versions"] if v.get("version")]
    elif lineage and lineage.get("versions"):
        versions = [v.get("version") for v in lineage["versions"] if v.get("version")]

    mode = os.getenv("MW_GOV_ACTION_MODE", "dryrun").lower()
    min_p = float(os.getenv("MW_GOV_ACTION_MIN_PRECISION", "0.75"))
    max_ece = float(os.getenv("MW_GOV_ACTION_MAX_ECE", "0.06"))

    actions: List[Dict[str, Any]] = []
    if not versions:
        actions = _seed_demo_actions()
    else:
        for v in trend.get("versions", []):
            ptrend = v.get("precision_trend", "stable")
            ece = abs(v.get("ece_delta", 0.03))
            action, reason, conf = _decide_action(ptrend, ece, min_p, max_ece)
            actions.append({
                "version": v.get("version"),
                "action": action,
                "confidence": conf,
                "reason": reason,
                "demo": False,
            })

    out_json = {
        "generated_at": _iso(_now_utc()),
        "mode": mode,
        "actions": actions,
        "demo": not bool(versions),
    }

    # Write JSON artifact
    jpath = models / "model_governance_actions.json"
    jpath.write_text(json.dumps(out_json, indent=2))

    # Optional plot (demo placeholder)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        ensure_dir(arts)
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=120)
        xs = list(range(len(actions)))
        colors = []
        for a in actions:
            if a["action"] == "promote":
                colors.append("green")
            elif a["action"] == "rollback":
                colors.append("red")
            else:
                colors.append("gray")
        ax.bar(xs, [a["confidence"] for a in actions], color=colors)
        ax.set_xticks(xs)
        ax.set_xticklabels([a["version"] for a in actions], rotation=30)
        ax.set_ylabel("Confidence")
        ax.set_title("Model Governance Actions")
        fig.tight_layout()
        fig.savefig(str(arts / "model_governance_actions.png"))
        plt.close(fig)
    except Exception:
        pass

    # Markdown CI block
    md.append("🧭 Model Governance Actions (72h)")
    for a in actions:
        line = f"{a['version']} → {a['action']} (confidence {a['confidence']}) [{', '.join(a['reason'])}]"
        md.append(line)
    md.append(f"mode: {mode}")
