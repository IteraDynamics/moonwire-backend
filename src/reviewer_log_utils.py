# src/reviewer_log_utils.py

import json
from pathlib import Path
from time import time
from typing import Any, Dict, List

from src.paths import (
    REVIEWER_IMPACT_LOG_PATH,
    REVIEWER_SCORES_PATH,
    RETRAINING_LOG_PATH,
)

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSON‐per‐line file and return a list of dicts.
    """
    entries: List[Dict[str, Any]] = []
    if path.exists():
        for line in path.open("r"):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries

def append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
    """
    Append a single dict as a JSON line to the given file.
    """
    # ensure parent dir exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

def get_reviewer_weight(reviewer_id: str) -> float:
    """
    Look up a reviewer's score in REVIEWER_SCORES_PATH,
    and map it to the same weight bands you use elsewhere:
      ≥ 0.75 → 1.25
      ≥ 0.5  → 1.0
      else   → 0.75
    """
    # default raw score = 0.0 if missing
    raw = 0.0
    if REVIEWER_SCORES_PATH.exists():
        for entry in load_jsonl(REVIEWER_SCORES_PATH):
            if entry.get("reviewer_id") == reviewer_id:
                raw = entry.get("score", 0.0)
                break

    if raw >= 0.75:
        return 1.25
    if raw >= 0.5:
        return 1.0
    return 0.75

def log_reviewer_action(payload: Dict[str, Any]) -> None:
    """
    Legacy entrypoint for your internal_router `/reviewer-impact-log`.
    Just appends the full payload (including any reviewer_weight you added)
    to the impact log.
    """
    append_jsonl(REVIEWER_IMPACT_LOG_PATH, payload)

def log_retrain_flag(payload: Dict[str, Any]) -> None:
    """
    Called by your flag-for-retraining router.
    Automatically stamps a timestamp and writes to RETRAINING_LOG_PATH.
    """
    record = payload.copy()
    record.setdefault("timestamp", time())
    append_jsonl(RETRAINING_LOG_PATH, record)

def apply_trust_delta(signal_id: str, delta: float) -> None:
    """
    Stub for actually applying a trust change to a signal.
    If you have an in-memory cache or DB, hook it here.
    For now, this is a no-op.
    """
    pass