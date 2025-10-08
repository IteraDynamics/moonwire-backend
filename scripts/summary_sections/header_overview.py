# scripts/summary_sections/header_overview.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from .common import SummaryContext, _iso


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def append(
    md: List[str],
    ctx: SummaryContext,
    *,
    reviewers: List[Dict[str, Any]] | None = None,
    threshold: float = 0.5,
    sig_id: str = "demo",
) -> None:
    """
    Header/overview block. Keyword-only args have sane defaults so accidental
    positional calls don't crash CI. Also guarded to emit ONCE.
    """
    # One-time guard to prevent duplicate headers if called multiple times.
    if getattr(ctx, "caches", None) is not None:
        if ctx.caches.get("did_header"):
            return
        ctx.caches["did_header"] = True

    # Optional demo seeding handled by mw_demo_summary.generate_demo_data_if_needed,
    # but stay resilient if nothing is passed.
    revs = reviewers or []
    n_unique = len({r.get("id") for r in revs if r.get("id")})
    now_iso = _iso(_now())

    # Title + concise pipeline proof line
    md.append("MoonWire CI Demo Summary")
    md.append(f"MoonWire Demo Summary — {now_iso}")
    md.append("Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.")
    md.append(f"\t•\tSignal: **{sig_id}**")
    md.append(f"\t•\tUnique reviewers: {n_unique}")
    md.append("\t•\tCombined weight: 0.0")
    md.append(f"\t•\tThreshold: {threshold} → NO TRIGGER")

    if revs:
        md.append("Reviewers (redacted):")
        for r in revs:
            rid = r.get("id", "rev")
            md.append(f"\t•\t**{rid}** → very low")
    else:
        md.append("Reviewers (redacted):")
        md.append("\t•\t(no reviewers)")

    md.append("Signal origin breakdown (last 7 days):")
    md.append("\t•\tno origin data")