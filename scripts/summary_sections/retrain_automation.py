# scripts/summary_sections/retrain_automation.py
# CI summary wrapper for Retrain Automation.
# Tries governance engine first; otherwise reads models/retrain_plan.json and renders.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .common import SummaryContext  # type: ignore
except Exception:  # pragma: no cover
    class SummaryContext:  # type: ignore
        def __init__(self, logs_dir: Path, models_dir: Path, is_demo: bool = False, artifacts_dir: Optional[Path] = None):
            self.logs_dir = logs_dir
            self.models_dir = models_dir
            self.artifacts_dir = artifacts_dir or (Path.cwd() / "artifacts")
            self.is_demo = is_demo


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return None


def _fmt_pp(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x*100:+.1f}pp"


def _fmt_delta(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.3f}"


def _call_governance_if_available(ctx: SummaryContext) -> Optional[Dict[str, Any]]:
    """
    If scripts.governance.retrain_automation is available, try to build a fresh plan.
    We support multiple function names to be resilient across branches.
    """
    try:
        from scripts.governance import retrain_automation as ra  # type: ignore
    except Exception:
        return None

    fn_names = [
        "build_retrain_plan",
        "build_plan",
        "run",
    ]
    for name in fn_names:
        fn = getattr(ra, name, None)
        if callable(fn):
            try:
                plan = fn(ctx)  # type: ignore[misc]
                if isinstance(plan, dict):
                    return plan
            except Exception:
                return None
    return None


def _render_markdown(md: List[str], plan: Dict[str, Any]) -> None:
    mode = plan.get("action_mode") or plan.get("mode") or "dryrun"
    demo = plan.get("demo", False)
    window_days = plan.get("window_days") or plan.get("window") or 30

    title = f"### 🔁 Retrain Automation ({window_days}d window) — mode: {mode}"
    if demo:
        title += " (demo)"
    md.append(title)

    cands = plan.get("candidates", [])
    if not cands:
        md.append("no retrain candidates")
        return

    for c in cands:
        origin = c.get("origin", "unknown")
        cur = c.get("current_version") or c.get("model_version") or "v?"
        newv = c.get("new_version") or c.get("proposed_version") or "v?"
        evald = c.get("eval", {})
        decision = c.get("decision", "plan")

        dp = _fmt_pp(evald.get("precision_delta"))
        de = _fmt_delta(evald.get("ece_delta"))
        df = _fmt_pp(evald.get("f1_delta"))

        left = f"{origin} {cur}"
        if decision == "promote" and newv:
            left = f"{origin} {cur} → {newv}"

        reasons = c.get("reason") or c.get("reasons") or []
        if isinstance(reasons, str):
            reasons = [reasons]

        right = [f"| ΔPrecision {dp} | ΔECE {de} | ΔF1 {df} [{decision}]"]
        if reasons:
            right.append(f"reason: {', '.join(reasons)}")

        md.append(f"{left:12s}  " + "  ".join(right))


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Public entrypoint used by the orchestrator (__init__.py).
    """
    models_dir = Path(getattr(ctx, "models_dir", Path("models")))
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", Path("artifacts")))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Try governance live computation first; then fall back to saved plan JSON.
    plan = _call_governance_if_available(ctx)
    if plan is None:
        plan = _read_json(models_dir / "retrain_plan.json")

    if not isinstance(plan, dict):
        md.append("\n> ⚠️ Retrain Automation: no plan available (module missing or plan file not found).\n")
        return

    _render_markdown(md, plan)