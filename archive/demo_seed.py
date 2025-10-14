# src/demo_seed.py

import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional

_ALLOWED_WEIGHTS = (0.75, 1.0, 1.25)

def seed_reviewers_if_empty(
    reviewers: List[Dict],
    now: Optional[datetime] = None,
    min_reviewers: int = 3,
    max_reviewers: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns (display_reviewers, seeded_events_for_timeline).

    - If reviewers is non-empty (real data present), returns (reviewers, []).
    - If empty, generates 3–5 mock reviewers with allowed weights and ISO timestamps
      within the last 60 minutes. Purely in-memory; NEVER writes to logs.
    """
    if reviewers:
        return reviewers, []

    now = now or datetime.now(timezone.utc)
    n = random.randint(min_reviewers, max_reviewers)
    events: List[Dict] = []
    for _ in range(n):
        events.append({
            "id": f"demo-{uuid.uuid4().hex[:8]}",
            "weight": random.choice(_ALLOWED_WEIGHTS),
            "timestamp": (now - timedelta(minutes=random.randint(1, 55))).isoformat(),
        })

    display = [{"id": e["id"], "weight": e["weight"]} for e in events]
    return display, events