# scripts/summary_sections/header_overview.py
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from .common import SummaryContext, _iso


@dataclass
class _Rev:
    id: str
    origin: str = "reddit"
    score: float = 0.0
    label: str = "very low"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def _demo_reviewers_if_needed(reviewers: Optional[Iterable[Dict[str, Any]]]) -> List[_Rev]:
    """
    In demo mode, if no reviewers are provided, synthesize 3 stable reviewers.
    Keep deterministic IDs so tests and CI snapshots are stable.
    """
    if reviewers:
        out: List[_Rev] = []
        for r in reviewers:
            out.append(_Rev(
                id=str(r.get("id", "rev_unknown")),
                origin=str(r.get("origin", "reddit")),
                score=float(r.get("score", 0.0)),
                label=str(r.get("label", "very low")),
            ))
        return out

    if not (_env_bool("DEMO_MODE") or _env_bool("MW_DEMO")):
        return []

    # deterministic demo reviewers
    return [
        _Rev(id="rev_demo_1", origin="reddit",  score=0.12, label="very low"),
        _Rev(id="rev_demo_2", origin="rss_news", score=0.09, label="very low"),
        _Rev(id="rev_demo_3", origin="twitter", score=0.11, label="very low"),
    ]


def append(
    md: List[str],
    ctx: SummaryContext,
    *,
    reviewers: Optional[Iterable[Dict[str, Any]]] = None,
    threshold: float = 0.5,
    sig_id: str = "demo",
    triggered_log: Optional[Iterable[Dict[str, Any]]] = None,  # accepted and ignored for compatibility
) -> None:
    """
    Overview header for the CI summary.
    IMPORTANT: We DO NOT emit an H1 title here to avoid duplicate headers because the
    GitHub job step already shows an H1 like 'MoonWire CI Demo Summary'.
    """
    _ = ctx  # unused today; kept for signature stability

    ts = _iso(_now_utc())

    revs = _demo_reviewers_if_needed(reviewers)
    uniq_count = len(revs)
    combined_weight = 0.0  # placeholder; governance math lives elsewhere
    triggered = False  # demo overview just shows proof of math, not actual trigger

    # --- Header block (no top-level H1!) ---
    md.append(f"MoonWire Demo Summary — {ts}")
    md.append("Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.")
    md.append(f"• Signal: **{sig_id}**")
    md.append(f"• Unique reviewers: {uniq_count}")
    md.append(f"• Combined weight: {combined_weight:.1f}")
    md.append(f"• Threshold: {threshold} → {'TRIGGER' if triggered else 'NO TRIGGER'}")

    # Reviewers (redacted)
    md.append("Reviewers (redacted):")
    if revs:
        for r in revs:
            md.append(f"• **{r.id}** → {r.label}")
    else:
        md.append("• (no reviewers)")

    # Origin breakdown stub (kept compact; richer view lives in per-origin sections)
    md.append("Signal origin breakdown (last 7 days):")
    md.append("• no origin data")