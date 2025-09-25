# scripts/mw_demo_summary.py
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple


def _ziso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


# -----------------------------------------------------------------------------
# Public API expected by tests
# -----------------------------------------------------------------------------
def generate_demo_data_if_needed(origins_rows: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Return (reviewers, events).

    Behavior:
      - If DEMO_MODE != "true" (case-insensitive), return ([], []).
      - If DEMO_MODE == "true":
          * If origins_rows is non-empty, return (origins_rows, <one event per reviewer>).
          * Otherwise, synthesize a tiny stable demo dataset and emit one event per reviewer.

    Always returns exactly two values to satisfy tests.
    """
    demo_mode = str(os.getenv("DEMO_MODE", "")).lower() == "true"
    if not demo_mode:
        return [], []

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    if origins_rows:
        reviewers = list(origins_rows)
        # one event per reviewer, aligned to reviewer timestamp if present
        events: List[Dict] = []
        for i, r in enumerate(reviewers):
            ts = r.get("timestamp")
            if not ts:
                ts = _ziso(now - timedelta(minutes=(len(reviewers) - 1 - i) * 5))
            events.append({
                "type": "demo_review_created",
                "at": ts,
                "meta": {"version": "v0.6.6", "note": "seeded in demo mode (passed-through)"},
                "review_id": r.get("id", f"rev_{i}"),
            })
        return reviewers, events

    # Seed a tiny deterministic demo set (3 reviewers)
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
    # Emit one event per reviewer so tests expecting equal lengths pass
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


# -----------------------------------------------------------------------------
# Minimal CLI for convenience in CI/manual runs
# -----------------------------------------------------------------------------
def _main() -> None:
    reviewers, events = generate_demo_data_if_needed([])
    demo = "true" if reviewers or events else "false"
    print(
        f"[mw_demo_summary] DEMO_MODE derived output present: {demo} "
        f"(reviewers={len(reviewers)}, events={len(events)})"
    )


if __name__ == "__main__":
    _main()