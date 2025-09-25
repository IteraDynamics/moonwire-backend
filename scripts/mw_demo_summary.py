# scripts/mw_demo_summary.py
from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Summary sections
from scripts.summary_sections.common import SummaryContext, ensure_dir, _iso
from scripts.summary_sections import (
    calibration_reliability_trend as cal_trend,  # market-aware overlays
    market_context,                              # BTC/ETH/SOL context
)

# -------------------------------------------------------------------
# Demo seeding (aligned with tests in tests/test_demo_seed.py)
# -------------------------------------------------------------------

def generate_demo_data_if_needed(reviewers_seed: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    If DEMO_MODE=true and no reviewers provided, seed 3 demo reviewers and
    emit one event per seeded reviewer. If DEMO_MODE=false, return inputs.
    If DEMO_MODE=true and reviewers are provided, pass-through with no events.

    Returns (reviewers, events).
    """
    demo_mode = str(os.getenv("DEMO_MODE", "")).lower() == "true"

    if not demo_mode:
        return reviewers_seed, []

    if reviewers_seed:
        # Demo on but real reviewers were supplied -> pass-through, no events
        return reviewers_seed, []

    now = datetime.now(timezone.utc)
    ts_floor = now.replace(minute=0, second=0, microsecond=0)

    reviewers = [
        {"id": "rev_demo_1", "origin": "reddit",  "score": 0.82, "timestamp": _iso(ts_floor)},
        {"id": "rev_demo_2", "origin": "rss_news","score": 0.74, "timestamp": _iso(ts_floor)},
        {"id": "rev_demo_3", "origin": "twitter", "score": 0.67, "timestamp": _iso(ts_floor)},
    ]

    # One event per reviewer (tests expect len(events) == len(reviewers))
    events = [
        {
            "type": "demo_review_created",
            "review_id": r["id"],
            "at": _iso(now),
            "meta": {"version": "v0.6.6", "note": "seeded in demo mode"},
        }
        for r in reviewers
    ]
    return reviewers, events

# -------------------------------------------------------------------
# Markdown summary builder
# -------------------------------------------------------------------

def _gather_plot_inventory(artifacts_dir: Path) -> List[str]:
    if not artifacts_dir.exists():
        return []
    return sorted([p.name for p in artifacts_dir.glob("*.png")])

def _append_section_safe(md: List[str], title: str, fn, ctx: SummaryContext):
    try:
        fn(md, ctx)
    except Exception as e:  # noqa: BLE001
        md.append(f"\n**{title}**: _error: {e!r}_\n")

def build_summary_md(ctx: SummaryContext) -> str:
    md: List[str] = []
    md.append("### MoonWire CI Demo Summary\n")

    # Market Context
    _append_section_safe(md, "Market Context", market_context.append, ctx)

    # Calibration & Reliability Trend (with market regime overlays)
    _append_section_safe(md, "Calibration & Reliability Trend", cal_trend.append, ctx)

    # Plots inventory
    plots = _gather_plot_inventory(Path(ctx.artifacts_dir))
    if plots:
        md.append("\n**🖼️ Plots available**")
        for name in plots:
            md.append(f"- {name}")
        md.append("")  # newline

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

    # Seed demo data if needed (no-op outside demo)
    _ = generate_demo_data_if_needed([])

    ctx = SummaryContext(
        logs_dir=logs,
        models_dir=models,
        is_demo=is_demo,
    )
    # Some sections expect artifacts_dir on ctx
    setattr(ctx, "artifacts_dir", artifacts)

    md = build_summary_md(ctx)
    (artifacts / "demo_summary.md").write_text(md)

if __name__ == "__main__":
    main()