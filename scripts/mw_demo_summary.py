# scripts/mw_demo_summary.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Summary sections
from scripts.summary_sections.common import SummaryContext, ensure_dir, _iso
from scripts.summary_sections import (
    calibration_reliability_trend as cal_trend,  # enhanced market-aware block
    market_context,                              # market context section
)

# -------------------------------------------------------------------
# Demo seeding (kept minimal; tested in tests/test_demo_seed.py)
# -------------------------------------------------------------------

def generate_demo_data_if_needed(reviewers_seed: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    If DEMO_MODE=true and no reviewers provided, seed 3 demo reviewers and a
    single 'demo_summary_generated' event. If DEMO_MODE=false, return inputs.

    Returns (reviewers, events).
    """
    demo_mode = str(os.getenv("DEMO_MODE", "")).lower() == "true"

    if not demo_mode:
        # Pass-through: no side-effects/events
        return reviewers_seed, []

    if reviewers_seed:
        # Respect given reviewers in demo mode; no synthetic summary event
        return reviewers_seed, []

    now = datetime.now(timezone.utc)
    reviewers = [
        {"id": "rev_demo_1", "origin": "reddit",  "score": 0.82, "timestamp": _iso(now.replace(minute=0))},
        {"id": "rev_demo_2", "origin": "rss_news","score": 0.74, "timestamp": _iso(now.replace(minute=0) + (now - now))},
        {"id": "rev_demo_3", "origin": "twitter", "score": 0.67, "timestamp": _iso(now.replace(minute=0))},
    ]
    events = [{
        "type": "demo_summary_generated",
        "at": _iso(now),
        "meta": {"version": "v0.6.6", "note": "seeded in demo mode"},
    }]
    return reviewers, events

# -------------------------------------------------------------------
# Markdown summary builder
# -------------------------------------------------------------------

def _gather_plot_inventory(artifacts_dir: Path) -> List[str]:
    if not artifacts_dir.exists():
        return []
    names = []
    for p in sorted(artifacts_dir.glob("*.png")):
        names.append(p.name)
    return names

def _append_section_safe(md: List[str], title: str, fn, ctx: SummaryContext):
    """Run a summary section, never crash summary, surface short error note."""
    try:
        fn(md, ctx)
    except Exception as e:  # noqa: BLE001
        md.append(f"\n**{title}**: _error: {e!r}_\n")

def build_summary_md(ctx: SummaryContext) -> str:
    md: List[str] = []
    md.append("### MoonWire CI Demo Summary\n")

    # --- Market Context (shows BTC/ETH/SOL and writes artifacts/models) ---
    _append_section_safe(md, "Market Context", market_context.append, ctx)

    # --- Calibration & Reliability Trend (market-aware overlays) ---
    _append_section_safe(md, "Calibration & Reliability Trend", cal_trend.append, ctx)

    # --- Plots inventory (so the CI summary always lists what’s viewable) ---
    plots = _gather_plot_inventory(Path(ctx.artifacts_dir))
    if plots:
        md.append("\n**🖼️ Plots available**")
        for name in plots:
            md.append(f"- {name}")
        md.append("")  # final newline

    return "\n".join(md)

# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main() -> None:
    repo_root = Path(os.getcwd())
    artifacts = ensure_dir(repo_root / "artifacts")
    models = ensure_dir(repo_root / "models")
    logs = ensure_dir(repo_root / "logs")

    is_demo = str(os.getenv("DEMO_MODE", "")).lower() == "true"

    # Seed (no-op unless demo + no reviewers provided)
    _ = generate_demo_data_if_needed([])

    ctx = SummaryContext(
        logs_dir=logs,
        models_dir=models,
        is_demo=is_demo,
    )
    # Some callers (and our enhanced sections) expect artifacts_dir on ctx
    setattr(ctx, "artifacts_dir", artifacts)

    md = build_summary_md(ctx)

    out_path = artifacts / "demo_summary.md"
    out_path.write_text(md)

if __name__ == "__main__":
    main()