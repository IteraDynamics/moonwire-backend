# scripts/summary_sections/drift_response.py
# CI summary wrapper for Automated Drift Response.
# Tries to call governance engine if available; otherwise reads the saved plan JSON.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

# Local common helpers are optional in some branches; keep imports minimal/safe.
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


def _fmt_delta_pp(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    # convert proportion deltas to percentage points with sign
    return f"{x*100:+.1f}pp"


def _fmt_delta_plain(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.3f}"


def _pick(s: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    out = {}
    for k in keys:
        if k in s:
            out[k] = s[k]
    return out


def _call_governance_if_available(ctx: SummaryContext) -> Optional[Dict[str, Any]]:
    """
    If scripts.governance.drift_response is available, try to build a fresh plan.
    We support multiple possible function names for forward/backward compatibility.
    """
    try:
        from scripts.governance import drift_response as dr  # type: ignore
    except Exception:
        return None

    fn_names = [
        "build_drift_response_plan",
        "build_plan",
        "run_policy",
    ]
    for name in fn_names:
        fn = getattr(dr, name, None)
        if callable(fn):
            try:
                # Prefer passing ctx; some variants expect only ctx, others allow (**kwargs)
                plan = fn(ctx)  # type: ignore[misc]
                if isinstance(plan, dict):
                    return plan
            except Exception:
                # If governance raised, fall back to file
                return None
    return None


def _render_markdown(md: List[str], plan: Dict[str, Any], demo_flag_from_file: Optional[bool] = None) -> None:
    action_mode = plan.get("action_mode") or plan.get("mode") or "dryrun"
    demo = plan.get("demo")
    if demo is None and demo_flag_from_file is not None:
        demo = demo_flag_from_file

    title = f"### 🛡️ Automated Drift Response (72h) — mode: {action_mode}"
    if demo:
        title += " (demo)"
    md.append(title)

    cands = plan.get("candidates", [])
    if not cands:
        md.append("no candidates detected")
        return

    for c in cands:
        origin = c.get("origin", "unknown")
        vers = c.get("model_version") or c.get("version") or c.get("current_version") or "v?"
        proposed = c.get("proposed_threshold")
        current = c.get("current_threshold")
        delta = c.get("delta")
        decision = c.get("decision") or "plan"
        reasons = c.get("reasons") or c.get("reason") or []
        if isinstance(reasons, str):
            reasons = [reasons]

        bt = c.get("backtest") or {}
        prec = _fmt_delta_pp(bt.get("precision_delta"))
        f1 = _fmt_delta_pp(bt.get("f1_delta"))
        ece = _fmt_delta_plain(bt.get("ece_delta"))

        left = f"{origin}/{vers}"
        right = []
        if proposed is not None and current is not None and delta is not None:
            right.append(f"→ {delta:+.2f} threshold")
        if bt:
            right.append(f"| ΔPrecision {prec} | ΔECE {ece} | ΔF1 {f1}")
        right.append(f"[{decision}]")
        if reasons:
            right.append(f"reason: {', '.join(reasons)}")

        md.append(f"{left:12s}  " + "  ".join(right))


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Public entrypoint used by the orchestrator (__init__.py).
    Produces a compact CI summary for drift response.
    """
    models_dir = Path(getattr(ctx, "models_dir", Path("models")))
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", Path("artifacts")))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Try to compute a fresh plan from governance; otherwise fall back to saved plan JSON.
    plan = _call_governance_if_available(ctx)
    plan_from_file = False
    if plan is None:
        plan_path = models_dir / "drift_response_plan.json"
        plan = _read_json(plan_path)
        plan_from_file = True

    # If still nothing, render an informative skip note
    if not isinstance(plan, dict):
        md.append("\n> ⚠️ Automated Drift Response: no plan available (module missing or plan file not found).\n")
        return

    # Render markdown
    _render_markdown(md, plan, demo_flag_from_file=plan.get("demo") if plan_from_file else None)