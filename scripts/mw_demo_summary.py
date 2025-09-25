# scripts/mw_demo_summary.py
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple

# -----------------------------------------------------------------------------
# Public API expected by tests
# -----------------------------------------------------------------------------

def generate_demo_data_if_needed(origins_rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Return (reviewers, events).

    Behavior:
      - If DEMO_MODE != "true" (case-insensitive), return ([], []).
      - If DEMO_MODE == "true":
          * If origins_rows is non-empty, just return (origins_rows, []) to be benign.
          * Otherwise, synthesize a tiny stable demo dataset.

    This function must ALWAYS return exactly two values to satisfy tests.
    """
    demo_mode = str(os.getenv("DEMO_MODE", "")).lower() == "true"
    if not demo_mode:
        # Explicitly return exactly two empty lists
        return [], []

    if origins_rows:
        # Preserve caller-provided rows and return no events
        return origins_rows, []

    # Seed a tiny deterministic demo set
    now = datetime.now(timezone.utc).replace(microsecond=0, second=0)
    reviewers = [
        {
            "id": "rev_demo_1",
            "origin": "reddit",
            "score": 0.82,
            "timestamp": (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
        },
        {
            "id": "rev_demo_2",
            "origin": "rss_news",
            "score": 0.31,
            "timestamp": (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        },
        {
            "id": "rev_demo_3",
            "origin": "twitter",
            "score": 0.67,
            "timestamp": now.isoformat().replace("+00:00", "Z"),
        },
    ]
    events = [
        {
            "type": "demo_summary_generated",
            "at": now.isoformat().replace("+00:00", "Z"),
            "meta": {"version": "v0.6.6", "note": "seeded in demo mode"},
        }
    ]
    return reviewers, events


# -----------------------------------------------------------------------------
# Minimal CLI so CI can call this module without side effects beyond printing.
# -----------------------------------------------------------------------------

def _main() -> None:
    """
    Kept intentionally minimal: in CI we mainly rely on other summary sections.
    This prints a short note so calling `python scripts/mw_demo_summary.py`
    doesn't fail, but it avoids writing files (tests don't require it here).
    """
    reviewers, events = generate_demo_data_if_needed([])
    demo = "true" if reviewers or events else "false"
    print(f"[mw_demo_summary] DEMO_MODE derived output present: {demo} "
          f"(reviewers={len(reviewers)}, events={len(events)})")


if __name__ == "__main__":
    _main()