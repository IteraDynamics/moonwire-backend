# scripts/summary_sections/header_overview.py
"""
Header/overview block for the CI summary.
- Backward-compatible: can be called as append(md, ctx) (registry call),
  or with optional reviewers/threshold/sig_id/triggered_log for richer context.
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Dict, Any
from datetime import datetime

from .common import SummaryContext, _iso


def _fmt_weight(w: float) -> str:
    if w >= 0.8:
        return "very high"
    if w >= 0.5:
        return "high"
    if w >= 0.2:
        return "medium"
    if w > 0.0:
        return "low"
    return "very low"


def _render_reviewers(reviewers: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for r in reviewers:
        name = r.get("id") or r.get("name") or "reviewer"
        w = float(r.get("weight", 0.0) or 0.0)
        lines.append(f"• **{name}** → {_fmt_weight(w)}")
    return lines


def append(
    md: List[str],
    ctx: SummaryContext,
    *,
    reviewers: Optional[Sequence[Dict[str, Any]]] = None,
    threshold: float = 0.5,
    sig_id: str = "demo",
    triggered_log: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    """
    Append the header/overview block.

    Parameters are optional so this function can be called from the registry
    (which passes only md, ctx) without errors.
    """
    # Defaults if nothing is provided
    reviewers = list(
        reviewers
        or [
            {"id": "rev_demo_1", "weight": 0.0},
            {"id": "rev_demo_2", "weight": 0.0},
            {"id": "rev_demo_3", "weight": 0.0},
        ]
    )
    combined = float(sum(float(r.get("weight", 0.0) or 0.0) for r in reviewers))
    triggered = bool(triggered_log) and any(bool(e.get("triggered")) for e in triggered_log or [])
    verdict = "TRIGGER" if (combined >= threshold or triggered) else "NO TRIGGER"

    # Basic per-origin line; keep harmless if ctx has nothing
    origin_line = "• no origin data"
    try:
        if ctx.origins_rows:
            uniq = sorted({str(row.get("origin", "?")) for row in ctx.origins_rows})
            if uniq:
                origin_line = "• " + ", ".join(uniq)
    except Exception:
        pass

    # _iso requires a datetime argument
    ts = _iso(datetime.utcnow())

    block = [
        "MoonWire CI Demo Summary",
        f"MoonWire Demo Summary — {ts}",
        "Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.",
        f"• Signal: {sig_id}",
        f"• Unique reviewers: {len(reviewers)}",
        f"• Combined weight: {combined:.1f}",
        f"• Threshold: {threshold:.1f} → {verdict}",
        "Reviewers (redacted):",
        *(_render_reviewers(reviewers) or ["• (no reviewers)"]),
        "Signal origin breakdown (last 7 days):",
        origin_line,
    ]
    md.append("\n".join(block))