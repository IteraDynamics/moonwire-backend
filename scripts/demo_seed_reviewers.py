# scripts/demo_seed_reviewers.py
import os
import json
import random
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone

LOGS_DIR = Path("logs")
RETRAINING_LOG = LOGS_DIR / "retraining_log.jsonl"
REVIEWER_SCORES = LOGS_DIR / "reviewer_scores.jsonl"

def _is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")

def _band_weight_from_score(score: float) -> float:
    if score is None:
        return 1.0
    if score >= 0.75:
        return 1.25
    if score >= 0.50:
        return 1.0
    return 0.75

def _append_jsonl(path: Path, obj: dict):
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")

def _load_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out

def seed_once(
    num_reviewers: int | None = None,
    signal_id: str | None = None,
) -> dict:
    """
    Seeds mock reviewers into logs/ when DEMO_MODE=true AND logs are empty or we
    explicitly want to add demo data. Returns a dict with metadata.

    When DEMO_MODE is false, this function does NOTHING and does not create directories/files.
    """
    # EARLY EXIT: do not touch filesystem if demo is off
    if not _is_demo_mode():
        print("DEMO_MODE is not enabled; skipping mock seed.")
        return {"seeded": False, "reason": "demo_off"}

    # From here on it's safe to touch the filesystem.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Decide whether to seed based on current state
    retrain_entries = _load_jsonl(RETRAINING_LOG)
    do_seed = False

    if not retrain_entries:
        # empty retraining log → we definitely want a seed
        do_seed = True

    # If a caller explicitly passed num_reviewers or signal_id, we seed anyway
    if num_reviewers is not None or signal_id is not None:
        do_seed = True

    if not do_seed:
        return {"seeded": False, "reason": "already_has_data"}

    # Defaults
    n = num_reviewers if (isinstance(num_reviewers, int) and 3 <= num_reviewers <= 7) else random.randint(3, 7)
    sid = signal_id or f"demo-sig-{uuid.uuid4().hex[:8]}"

    # Generate reviewer_scores if missing (or just append fresh snapshots)
    now = datetime.now(timezone.utc)
    reviewers = []
    for _ in range(n):
        rid = f"demo-{uuid.uuid4().hex[:8]}"
        score = round(random.uniform(0.30, 1.00), 2)
        weight = _band_weight_from_score(score)
        reviewers.append({"reviewer_id": rid, "score": score, "weight": weight})

    # Write reviewer_scores snapshots
    for r in reviewers:
        _append_jsonl(REVIEWER_SCORES, {
            "reviewer_id": r["reviewer_id"],
            "score": r["score"],
            "timestamp": now.isoformat()
        })

    # Write retraining flags (first flag counts per reviewer, but we just seed one each)
    for r in reviewers:
        # Spread timestamps over last 24h
        ts = now - timedelta(seconds=random.randint(60, 24 * 3600))
        _append_jsonl(RETRAINING_LOG, {
            "signal_id": sid,
            "reviewer_id": r["reviewer_id"],
            "reviewer_weight": r["weight"],  # include explicit weight for clarity
            "timestamp": ts.timestamp(),     # numeric epoch for consistency
        })

    return {
        "seeded": True,
        "signal_id": sid,
        "num_reviewers": n,
    }

if __name__ == "__main__":
    out = seed_once()
    print(json.dumps(out, indent=2))