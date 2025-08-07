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

# ─── In-Memory Trust Store ──────────────────────────────────────────────────────
# Maintains current trust for each signal for the lifetime of the process/tests.
TRUST_SCORES: Dict[str, float] = {}

def get_signal_trust(signal_id: str) -> float:
    return TRUST_SCORES.get(signal_id, 0.0)

def apply_trust_delta(signal_id: str, delta: float) -> None:
    old = TRUST_SCORES.get(signal_id, 0.0)
    TRUST_SCORES[signal_id] = old + delta
# ────────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")

def get_reviewer_weight(reviewer_id: str) -> float:
    raw = 0.0
    if REVIEWER_SCORES_PATH.exists():
        for e in load_jsonl(REVIEWER_SCORES_PATH):
            if e.get("reviewer_id") == reviewer_id:
                raw = e.get("score", 0.0)
                break
    if raw >= 0.75:
        return 1.25
    if raw >= 0.5:
        return 1.0
    return 0.75

def log_reviewer_action(payload: Dict[str, Any]) -> None:
    append_jsonl(REVIEWER_IMPACT_LOG_PATH, payload)

def log_retrain_flag(payload: Dict[str, Any]) -> None:
    record = payload.copy()
    record.setdefault("timestamp", time())
    append_jsonl(RETRAINING_LOG_PATH, record)