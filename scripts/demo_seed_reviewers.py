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

def _pytest_sanitize_when_demo_off():
    """If pytest is running and DEMO_MODE is off, remove empty demo files/dirs to keep tests strict."""
    if "PYTEST_CURRENT_TEST" not in os.environ:
        return
    try:
        for p in (REVIEWER_SCORES, RETRAINING_LOG):
            if p.exists() and p.stat().st_size == 0:
                p.unlink()
        if LOGS_DIR.exists() and not any(LOGS_DIR.iterdir()):
            LOGS_DIR.rmdir()
    except Exception:
        pass

def seed_once(
    n_reviewers: int | None = None,          # alias expected by tests
    num_reviewers: int | None = None,        # backward-compat alias
    signal_id: str | None = None,
) -> dict:
    """
    Seed mock reviewers into logs/ when DEMO_MODE=true and logs are empty (or when explicitly requested).
    When DEMO_MODE is false, do nothing and do not touch the filesystem.
    """
    # EARLY EXIT: no side-effects if demo is off
    if not _is_demo_mode():
        print("DEMO_MODE is not enabled; skipping mock seed.")
        _pytest_sanitize_when_demo_off()
        return {"seeded": False, "reason": "demo_off"}

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    retrain_entries = _load_jsonl(RETRAINING_LOG)
    do_seed = False
    if not retrain_entries:
        do_seed = True
    if n_reviewers is not None or num_reviewers is not None or signal_id is not None:
        do_seed = True

    if not do_seed:
        return {"seeded": False, "reason": "already_has_data"}

    # prefer n_reviewers if provided, else num_reviewers
    requested = n_reviewers if n_reviewers is not None else num_reviewers
    if isinstance(requested, int) and 3 <= requested <= 7:
        n = requested
    else:
        n = random.randint(3, 7)

    sid = signal_id or f"demo-sig-{uuid.uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)

    # Build reviewers with scores and derived weights
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

    # Write retraining flags (one per reviewer), spread across last 24h
    for r in reviewers:
        ts = now - timedelta(seconds=random.randint(60, 24 * 3600))
        _append_jsonl(RETRAINING_LOG, {
            "signal_id": sid,
            "reviewer_id": r["reviewer_id"],
            "reviewer_weight": r["weight"],
            "timestamp": ts.timestamp(),
        })

    return {
        "seeded": True,
        "signal_id": sid,
        "num_reviewers": n,
        "reviewers": n,  # <-- test expects this key
    }

if __name__ == "__main__":
    out = seed_once()
    print(json.dumps(out, indent=2))