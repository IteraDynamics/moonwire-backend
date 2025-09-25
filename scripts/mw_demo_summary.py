# scripts/mw_demo_summary.py
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple


def _ziso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def generate_demo_data_if_needed(origins_rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Return (reviewers, events).

    Contract required by tests:
      - If DEMO_MODE != "true": return ([], []).
      - If DEMO_MODE == "true" and `origins_rows` is non-empty (real reviewers): return (origins_rows, []).
      - If DEMO_MODE == "true" and no reviewers provided: synthesize 3 reviewers and
        emit one event per reviewer (len(events) == len(reviewers)).
    """
    demo_mode = str(os.getenv("DEMO_MODE", "")).lower() == "true"
    if not demo_mode:
        return [], []

    if origins_rows:
        # In demo mode but real reviewers were passed in -> do not synthesize events.
        return list(origins_rows), []

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    reviewers = [
        {
            "id": "rev_demo_1",
            "origin": "reddit",
            "score": 0.82,
            "timestamp": _ziso(now - timedelta(hours=2)),
        },
        {
            "id": "rev_demo_2",
            "origin": "rss_news",
            "score": 0.31,
            "timestamp": _ziso(now - timedelta(hours=1)),
        },
        {
            "id": "rev_demo_3",
            "origin": "twitter",
            "score": 0.67,
            "timestamp": _ziso(now),
        },
    ]
    events = [
        {
            "type": "demo_review_created",
            "at": r["timestamp"],
            "meta": {"version": "v0.6.6", "note": "seeded in demo mode"},
            "review_id": r["id"],
        }
        for r in reviewers
    ]
    return reviewers, events


def _main() -> None:
    reviewers, events = generate_demo_data_if_needed([])
    demo = "true" if reviewers or events else "false"
    print(
        f"[mw_demo_summary] DEMO_MODE derived output present: {demo} "
        f"(reviewers={len(reviewers)}, events={len(events)})"
    )


if __name__ == "__main__":
    _main()