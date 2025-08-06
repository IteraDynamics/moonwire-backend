import json
import time
from pathlib import Path
from typing import Any, Dict, List

from src.paths import REVIEWER_IMPACT_LOG_PATH, REVIEWER_SCORES_PATH

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read every line of JSONL at `path` into a list of dicts.
    Returns an empty list if the file does not exist.
    """
    if not path.exists():
        return []
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """
    Append a JSON-serializable object as one line to `path`.
    Ensures parent directories exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")

def get_reviewer_weight(reviewer_id: str) -> float:
    """
    Look up a reviewer’s latest score in reviewer_scores.jsonl 
    and return the corresponding weight multiplier:
      • score >= 0.75 → 1.25
      • 0.5 <= score < 0.75 → 1.0
      • score < 0.5 → 0.75
    Defaults to 1.0 if no record is found.
    """
    records = load_jsonl(REVIEWER_SCORES_PATH)
    for rec in records:
        if rec.get("reviewer_id") == reviewer_id:
            score = rec.get("score", 0.0)
            if score >= 0.75:
                return 1.25
            if score >= 0.5:
                return 1.0
            return 0.75
    return 1.0

def apply_trust_delta(signal_id: str, delta: float) -> float:
    """
    Stub for applying a trust-delta to a signal’s stored trust score.
    Should return the previous trust score before applying the delta.
    You’ll need to hook this into your actual trust-store.
    """
    # TODO: read the signal’s current trust from your datastore,
    #       apply `delta`, persist the new score, then return the old one.
    # For now we return 0.0 as the “previous” score:
    return 0.0