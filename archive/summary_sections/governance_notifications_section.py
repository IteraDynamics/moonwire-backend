# scripts/summary_sections/governance_notifications_section.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import SummaryContext

def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        pass
    return None


def _fmt_delta(ev: Dict[str, Any]) -> str:
    d = ev.get("delta") or {}
    parts = []
    if "F1" in d and isinstance(d["F1"], (int, float)):
        parts.append(f"ΔF1 {d['F1']:+.02f}")
    if "ECE" in d and isinstance(d["ECE"], (int, float)):
        parts.append(f"ΔECE {d['ECE']:+.02f}")
    return ", ".join(parts)


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    CI Markdown block renderer for Multi-Tier Governance Notifications.
    """
    models = Path(ctx.models_dir or "models")
    j = _read_json(models / "governance_notifications_digest.json")
    if not j:
        md.append("\n> ⚠️ Notifications (no digest found)")
        return

    mode = j.get("mode", "on")
    routing = j.get("routing", {})
    crit = j.get("critical", []) or []
    info = j.get("info", []) or []
    run_url = j.get("run_url")

    md.append("📣 Notifications (72h)")

    if crit:
        first = True
        for ev in crit:
            line = f"• {ev.get('version','?')} → {ev.get('type','?')}"
            dtxt = _fmt_delta(ev)
            if dtxt:
                line += f" ({dtxt}"
                if isinstance(ev.get("conf"), (int, float)):
                    line += f", conf {ev['conf']:.2f})"
                else:
                    line += ")"
            elif isinstance(ev.get("conf"), (int, float)):
                line += f" (conf {ev['conf']:.2f})"
            if first:
                line += f" → routed: {routing.get('critical','print')}"
                first = False
            md.append(line)
    else:
        md.append("• critical (0)")

    if info:
        first = True
        for ev in info:
            line = f"• {ev.get('version','?')} → {ev.get('type','info')}"
            if isinstance(ev.get("conf"), (int, float)):
                line += f" (conf {ev['conf']:.2f})"
            if first:
                line += f" → routed: {routing.get('summary','print')}"
                first = False
            md.append(line)
    else:
        md.append("• info (0)")

    links_line = "links: artifacts attached"
    if run_url:
        links_line = "links: CI run / artifacts attached"
    md.append(links_line)

    md.append(f"mode: {mode} | critical→{routing.get('critical','print')}, summary→{routing.get('summary','print')}")