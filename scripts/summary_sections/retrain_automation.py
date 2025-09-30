"""
Summary renderer for Retrain Automation.

Reads models/retrain_plan.json (written by scripts/governance/retrain_automation.py)
and emits a compact CI block. Demo-aware. Never raises.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Any, Dict
import json
import os

def _read(path: Path) -> Dict[str,Any]:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception:
        pass
    return {}

def append(md: List[str], ctx) -> None:
    models_dir = Path(getattr(ctx, "models_dir", "models"))
    plan = _read(models_dir / "retrain_plan.json")
    if not plan:
        md.append("⚠️ Retrain Automation: no plan available (module missing or plan file not found).")
        return

    demo_tag = " (demo)" if plan.get("demo") else ""
    mode = plan.get("action_mode","dryrun")
    md.append(f"🔁 Retrain Automation (30d window) — mode: {mode}{demo_tag}")
    cands = plan.get("candidates") or []
    if not cands:
        md.append("no candidates detected")
        return

    lines: List[str] = []
    for c in cands:
        origin = c.get("origin","unknown")
        cv = c.get("current_version","v0")
        nv = c.get("new_version") or "tbd"
        evald = c.get("eval") or {}
        pd = evald.get("precision_delta", 0.0)
        ed = evald.get("ece_delta", 0.0)
        fd = evald.get("f1_delta", 0.0)
        decision = c.get("decision","hold")
        lines.append(f"{origin} {cv} → {nv}  | ΔP {pd:+.3f} | ΔECE {ed:+.3f} | ΔF1 {fd:+.3f} [{decision}]")

    md.extend(lines)