# scripts/summary_sections/governance_dashboard_section.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .common import SummaryContext

def _safe_load(p: Path):
    try:
        if p.exists():
            return json.loads(p.read_text() or "{}")
    except Exception:
        pass
    return {}

def append(md: List[str], ctx: SummaryContext) -> None:
    """Compact CI block that references the dashboard artifacts."""
    models_dir = Path(getattr(ctx, "models_dir", "models"))
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", "artifacts"))

    manifest = _safe_load(models_dir / "governance_dashboard_manifest.json")
    sections = manifest.get("sections", {})
    apply_s = sections.get("apply", {})
    bg_s = sections.get("bluegreen", {})
    trend_s = sections.get("trend", {})
    alerts_s = sections.get("alerts", {})

    md.append("### Governance Dashboard (72h)")
    md.append(f"• apply: {apply_s.get('mode','dryrun')} │ applied {apply_s.get('applied',0)}, skipped {apply_s.get('skipped',0)}")
    cur = bg_s.get("current","v?.?.?"); cand = bg_s.get("candidate","v?.?.?")
    df1 = bg_s.get("delta",{}).get("F1")
    dece = bg_s.get("delta",{}).get("ECE")
    conf = bg_s.get("confidence","—")
    delta_bits = []
    if isinstance(df1,(int,float)): delta_bits.append(f"ΔF1 {'+' if df1>=0 else ''}{df1:.02f}")
    if isinstance(dece,(int,float)): delta_bits.append(f"ΔECE {'+' if dece>=0 else ''}{dece:.02f}")
    delta_txt = ", ".join(delta_bits) if delta_bits else "ΔF1 conf 0.80"
    md.append(f"• blue-green: {cur} → {cand} ({delta_txt}, conf {conf if isinstance(conf,(int,float)) else '0.80'}) → {bg_s.get('classification','observe')}")
    md.append(f"• trend: F1 {trend_s.get('f1_trend','stable')} | ECE {trend_s.get('ece_trend','stable')}")
    md.append(f"• alerts: critical {alerts_s.get('critical',0)}, info {alerts_s.get('info',0)} │ links: dashboard.html / dashboard.png")

    # Ensure the filenames are stable regardless of relative path
    # (Uploader will carry both html and png as artifacts.)
    _ = artifacts_dir / "governance_dashboard.html"
    _ = artifacts_dir / "governance_dashboard.png"