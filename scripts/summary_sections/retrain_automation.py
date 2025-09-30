# scripts/summary_sections/retrain_automation.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_json(p: Path, default=None):
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return default


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fmt_pp(x: float) -> str:
    """Format a delta as percentage points with sign, e.g., +1.8pp."""
    try:
        return f"{x*100:+.1f}pp" if abs(x) < 1.0 else f"{x:+.3f}"  # tolerate already-in-fraction inputs
    except Exception:
        return f"{x}"


def append(md: List[str], ctx) -> None:
    """
    Orchestrate retrain automation and render a concise CI summary block.
    Robust to missing artifacts; in demo mode, seeds a friendly preview (via governance module).
    """
    models_dir = Path(getattr(ctx, "models_dir", "models"))
    logs_dir = Path(getattr(ctx, "logs_dir", "logs"))
    _ensure_dir(models_dir)
    _ensure_dir(logs_dir)

    # Run governance orchestrator
    try:
        from scripts.governance import retrain_automation as ra
        plan = ra.run(ctx)  # writes/updates models/retrain_plan.json internally
    except Exception as e:
        md.append(f"\n> ❌ Retrain Automation failed: `{type(e).__name__}: {e}`\n")
        return

    # Render markdown
    mode = (plan.get("action_mode") or os.getenv("MW_RETRAIN_ACTION", "dryrun")).lower()
    demo_tag = " (demo)" if plan.get("demo") else ""
    md.append(f"### 🔁 Retrain Automation (30d window) — mode: {mode}{demo_tag}")

    cands = plan.get("candidates", [])
    if not cands:
        md.append("no candidates")
        return

    for c in cands:
        origin = c.get("origin", "unknown")
        currver = c.get("current_version", "v0")
        newver = c.get("new_version") or f"v{int(currver.lstrip('v') or '0') + 1}"
        evald = c.get("eval", {}) or {}
        prec_delta = evald.get("precision_delta")
        ece_delta = evald.get("ece_delta")
        f1_delta = evald.get("f1_delta")
        decision = c.get("decision", "plan")

        # Format line based on decision + available eval
        if prec_delta is None and ece_delta is None and f1_delta is None:
            # planning only (no eval yet)
            reasons = ", ".join(c.get("reasons", []) or ["plan"])
            md.append(f"{origin} {currver} → {newver}  | plan              | reason: {reasons}")
        else:
            # have eval deltas
            md.append(
                f"{origin} {currver} → {newver}  | ΔPrecision {_fmt_pp(prec_delta)} | ΔECE {ece_delta:+.3f} | ΔF1 {_fmt_pp(f1_delta)} "
                f"[{'promote' if decision == 'promote' else 'hold'}]"
            )

    # Footer guardrails note (mirrors prompt)
    md.append(
        "Guardrails: min_labels≥300, cooldown 24h, promotion requires precision↑ & ECE↓ (≥0.01). All actions logged."
    )
