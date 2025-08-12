# scripts/demo_seed_reviewers.py
#!/usr/bin/env python3
"""
Seeds realistic mock reviewer activity into logs/ for demo mode.

- Generates 3–7 reviewers with scores in [0.30, 1.00]
- Appends reviewer records to logs/reviewer_scores.jsonl
- Appends flags to logs/retraining_log.jsonl for one recent signal
- Never writes unless DEMO_MODE=true
"""

import json
import os
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.demo_mode import is_demo_mode

LOGS_DIR = Path("logs")
SCORES_PATH = LOGS_DIR / "reviewer_scores.jsonl"
RETRAINING_LOG_PATH = LOGS_DIR / "retraining_log.jsonl"


def _band_weight_from_score(score: float) -> float:
    """Kept for parity with your endpoints; not written in flags (we rely on fallback)."""
    if score >= 0.75:
        return 1.25
    if score >= 0.50:
        return 1.0
    return 0.75


def _write_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def seed_once(n_reviewers: int | None = None, signal_id: str | None = None) -> dict:
    """
    Seed a single signal with mock reviewers and recent flags.
    Returns a small dict for debugging / tests.
    """
    if not is_demo_mode():
        print("DEMO_MODE is not enabled; skipping mock seed.")
        return {"seeded": False, "reviewers": 0, "signal_id": None}

    n = n_reviewers or random.randint(3, 7)
    sig = signal_id or f"demo-{uuid.uuid4().hex[:8]}"

    # Generate reviewers & scores
    reviewers: list[tuple[str, float]] = []
    for _ in range(n):
        rid = f"r-{uuid.uuid4().hex[:6]}"
        score = round(random.uniform(0.30, 1.00), 2)
        reviewers.append((rid, score))

    # Append reviewer scores
    now_iso = datetime.now(timezone.utc).isoformat()
    for rid, score in reviewers:
        _write_jsonl(
            SCORES_PATH,
            {
                "reviewer_id": rid,
                "score": score,
                "last_updated": now_iso,
            },
        )

    # Append flags in the last 24h; one primary per reviewer + 0–2 duplicates
    now = datetime.now(timezone.utc)
    for rid, _score in reviewers:
        base_ts = now - timedelta(seconds=random.randint(60, 24 * 3600))
        _write_jsonl(
            RETRAINING_LOG_PATH,
            {
                "signal_id": sig,
                "reviewer_id": rid,
                # epoch seconds keeps compatibility with existing endpoints/tests
                "timestamp": base_ts.timestamp(),
                # omit "reviewer_weight" to exercise banded fallback in endpoints
            },
        )
        for _ in range(random.randint(0, 2)):
            dup_ts = base_ts + timedelta(minutes=random.randint(5, 240))
            _write_jsonl(
                RETRAINING_LOG_PATH,
                {
                    "signal_id": sig,
                    "reviewer_id": rid,
                    "timestamp": dup_ts.timestamp(),
                },
            )

    print(f"Seeded DEMO reviewers: {n} for signal {sig}")
    return {"seeded": True, "reviewers": n, "signal_id": sig}


if __name__ == "__main__":
    seed_once()
