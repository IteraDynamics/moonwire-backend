# src/consensus_router.py

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import time

from src.paths import (
    RETRAINING_LOG_PATH,
    REVIEWER_SCORES_PATH,
    RETRAINING_TRIGGERED_LOG_PATH,
)

router = APIRouter(prefix="/internal")

class EvaluateRequest(BaseModel):
    signal_id: str

DEFAULT_THRESHOLD = 2.5

def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_consensus_status(signal_id: str, raise_on_empty: bool = False):
    entries = load_jsonl(RETRAINING_LOG_PATH)
    matching = [e for e in entries if e.get("signal_id") == signal_id]
    if not matching:
        if raise_on_empty:
            raise HTTPException(404, detail="No entries for this signal_id")
        return {"signal_id": signal_id, "total_reviewers": 0, "combined_weight": 0.0, "reviewers": []}

    seen = set()
    deduped = []
    for entry in matching:
        rid = entry["reviewer_id"]
        if rid not in seen:
            seen.add(rid)
            deduped.append(entry)

    scores = {}
    for s in load_jsonl(REVIEWER_SCORES_PATH):
        scores[s["reviewer_id"]] = s.get("score", 1.0)

    def compute_weight(entry):
        if "reviewer_weight" in entry:
            return entry["reviewer_weight"]
        score = scores.get(entry["reviewer_id"])
        if score is None:
            return 1.0
        if score >= 0.75:
            return 1.25
        elif score >= 0.5:
            return 1.0
        else:
            return 0.75

    reviewers = []
    total_weight = 0.0
    for entry in deduped:
        weight = compute_weight(entry)
        reviewers.append({
            "reviewer_id": entry["reviewer_id"],
            "weight": weight
        })
        total_weight += weight

    return {
        "signal_id": signal_id,
        "total_reviewers": len(reviewers),
        "combined_weight": total_weight,
        "reviewers": reviewers
    }

@router.post("/evaluate-consensus-retraining")
def evaluate_consensus(req: EvaluateRequest):
    threshold = DEFAULT_THRESHOLD
    consensus = get_consensus_status(req.signal_id, raise_on_empty=False)

    total_weight = consensus["combined_weight"]
    reviewers = consensus["reviewers"]
    triggered = total_weight >= threshold

    if triggered:
        log_entry = {
            "signal_id": req.signal_id,
            "total_weight": total_weight,
            "threshold": threshold,
            "reviewers": reviewers,
            "timestamp": time.time(),
        }
        with open(RETRAINING_TRIGGERED_LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return {
        "triggered": triggered,
        "total_weight": total_weight,
        "reviewers": reviewers
    }

@router.get("/consensus-status/{signal_id}", status_code=200)
def get_consensus_status_route(signal_id: str):
    return get_consensus_status(signal_id, raise_on_empty=True)
