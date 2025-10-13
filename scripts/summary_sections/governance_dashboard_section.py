# scripts/summary_sections/governance_dashboard_section.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.summary_sections.common import SummaryContext, ensure_dir
# Soft import builder so section can self-heal if caller forgot to run it
def _try_build(ctx: SummaryContext) -> Optional[Dict[str, Any]]:
    try:
        from scripts.dashboard.governance_dashboard import build_dashboard  # type: ignore
        return build_dashboard(ctx)
    except Exception:
        return None

def _load_manifest(models_dir: Path) -> Optional[Dict[str, Any]]:
    path = models_dir / "governance_dashboard_manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text() or "{}")
        except Exception:
            return None
    return None

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Append the compact Governance Dashboard block using manifest.
    If the manifest is missing, we try to build it on the fly (best-effort).
    """
    models_dir = Path(ctx.models_dir)
    manifest = _load_manifest(models_dir)
    if not manifest:
        manifest = _try_build(ctx) or _load_manifest(models_dir) or {}

    secs = manifest.get("sections", {})
    apply_s = secs.get("apply", {})
    bg_s = secs.get("bluegreen", {})
    trend_s = secs.get("trend", {})
    alerts_s = secs.get("alerts", {})
    window_h = manifest.get("window_hours", 72)

    # Compose lines (compact)
    md.append(f"🧭 Governance Dashboard ({window_h}h)")
    md.append(
        f"• apply: {apply_s.get('mode','dryrun')} | applied {apply_s.get('applied',0)}, skipped {apply_s.get('skipped',0)}"
    )
    md.append(
        "• blue-green: "
        f"{bg_s.get('current','?')} → {bg_s.get('candidate','?')} "
        f"(ΔF1 {_fmt(bg_s.get('confidence'), bg_s.get('classification'), manifest, use_deltas=True)}) "
        f"→ {bg_s.get('classification','observe')}"
    )
    # Replace parenthetical with explicit deltas if available in bluegreen_promotion.json (not in manifest)
    # Keep compact here; deltas are present in dashboard HTML.

    md.append(
        f"• trend: F1 {trend_s.get('f1_trend','stable')} | ECE {trend_s.get('ece_trend','stable')}"
    )
    md.append(
        f"• alerts: critical {alerts_s.get('critical',0)}, info {alerts_s.get('info',0)} | "
        "links: dashboard.html / dashboard.png"
    )

def _fmt(conf: Any, cls: Any, manifest: Dict[str, Any], use_deltas: bool = True) -> str:
    """
    Small helper to place confidence nicely; when `use_deltas` is True we only render confidence,
    since Δs are detailed in the HTML dashboard.
    """
    try:
        c = float(conf)
        return f"conf {c:.2f}"
    except Exception:
        return "conf n/a"