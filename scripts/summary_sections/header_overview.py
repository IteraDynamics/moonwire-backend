# scripts/summary_sections/header_overview.py
from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from .common import (
    SummaryContext,
    red,
    weight_to_label,
)

def append(
    md: List[str],
    ctx: SummaryContext,
    *,
    reviewers: List[Dict[str, Any]],
    threshold: float,
    sig_id: str,
    triggered_log: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Writes the summary header, reviewer list, and origin breakdown.
    NOTE: We intentionally do NOT write a top-level '# MoonWire CI Demo Summary' H1
    to avoid duplicating the job's heading in the GitHub summary step.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Totals
    total_weight = round(sum(r.get("weight", 0.0) for r in reviewers), 2)
    would_trigger = total_weight >= float(threshold)

    # Last retrain trigger time for the same signal_id (best-effort)
    last_trig = None
    if triggered_log:
        try:
            last_trig = max(
                (t for t in triggered_log if t.get("signal_id") == sig_id),
                key=lambda x: x.get("timestamp", 0),
                default=None,
            )
        except Exception:
            last_trig = None

    # ---- Header line (no extra H1) ----
    md.append(f"MoonWire Demo Summary — {now_iso}")
    md.append("Pipeline proof (CI): end-to-end tests passed; consensus math reproduced on latest flagged signal.")
    md.append(f"- **Signal:** `{red(sig_id)}`")
    md.append(f"- **Unique reviewers:** {len(reviewers)}")
    md.append(f"- **Combined weight:** **{total_weight}**")
    md.append(f"- **Threshold:** **{threshold}** → **{'TRIGGERS' if would_trigger else 'NO TRIGGER'}**")
    if last_trig:
        md.append(f"- **Last retrain trigger logged:** {last_trig.get('timestamp','')}")

    # ---- Reviewers (hashed/redacted) ----
    md.append("\n**Reviewers (redacted):**")
    if reviewers:
        for r in reviewers:
            rid = r.get("id", "")
            w = float(r.get("weight", 0.0))
            md.append(f"- `{red(rid)}` → {weight_to_label(w)}")
    else:
        md.append("- _none found in this run_")

    # ---- Origin breakdown (from ctx.origins_rows) ----
    md.append("\n**Signal origin breakdown (last 7 days):**")
    rows = ctx.origins_rows or []
    if rows:
        for o in rows:
            origin = o.get("origin", "unknown")
            cnt = o.get("count", 0)
            pct = o.get("percent")
            # Format percent whether it's numeric or pre-formatted
            try:
                if pct is None:
                    pct_s = "0%"
                elif isinstance(pct, (int, float)):
                    pct_s = f"{float(pct):.1f}%"
                else:
                    pct_s = str(pct)
            except Exception:
                pct_s = "n/a"
            md.append(f"- {origin}: {cnt} ({pct_s})")
    else:
        md.append("- _no origin data_")