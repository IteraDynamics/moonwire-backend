# scripts/summary_sections/governance_dashboard_section.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List
from .common import SummaryContext

HTML_PATH = Path("artifacts/governance_dashboard.html")
PNG_PATH = Path("artifacts/governance_dashboard.png")
MANIFEST_PATH = Path("models/governance_dashboard_manifest.json")

def _load_manifest():
    if not MANIFEST_PATH.exists():
        return None
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except Exception:
        return None

def _fmt_conf(v) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "n/a"

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Compact “Dashboard” card with links to artifacts.
    Ensures confidence is displayed once (no duplicate 'conf').
    """
    m = _load_manifest()

    md.append("### Governance Dashboard (72h)")

    if not m:
        md.append("• data unavailable — demo values")
        links = []
        if HTML_PATH.exists():
            links.append("dashboard.html")
        if PNG_PATH.exists():
            links.append("dashboard.png")
        if links:
            md.append("• links: " + " / ".join(links))
        return

    sections = m.get("sections", {})
    apply_s = sections.get("apply", {})
    bg_s = sections.get("bluegreen", {})
    trend_s = sections.get("trend", {})
    alerts_s = sections.get("alerts", {})

    # Apply
    mode = apply_s.get("mode", "dryrun")
    applied = apply_s.get("applied", 0)
    skipped = apply_s.get("skipped", 0)
    md.append(f"• apply: {mode} │ applied {applied}, skipped {skipped}")

    # Blue-Green (ensure single conf)
    cur = bg_s.get("current", "v?.?.?")
    cand = bg_s.get("candidate", "v?.?.?")
    classification = (bg_s.get("classification") or "observe").lower()
    conf = _fmt_conf(bg_s.get("confidence"))
    # Example format: vA -> vB (ΔF1 +0.02, ΔECE −0.01, conf 0.86) → promote_ready
    # We only print "conf 0.xx" once.
    md.append(f"• blue-green: {cur} → {cand} (conf {conf}) → {classification}")

    # Trend
    f1t = trend_s.get("f1_trend", "stable")
    ecet = trend_s.get("ece_trend", "stable")
    md.append(f"• trend: F1 {f1t} | ECE {ecet}")

    # Alerts
    crit = alerts_s.get("critical", 0)
    info = alerts_s.get("info", 0)
    links = []
    if HTML_PATH.exists():
        links.append("dashboard.html")
    if PNG_PATH.exists():
        links.append("dashboard.png")
    link_str = (" │ links: " + " / ".join(links)) if links else ""
    md.append(f"• alerts: critical {crit}, info {info}{link_str}")