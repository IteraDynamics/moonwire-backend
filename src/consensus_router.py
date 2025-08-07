# src/consensus_router.py

import json
import time
from typing import List, Dict
from pathlib import Path
from fastapi import APIRouter, HTTPException, Body

from src.paths import RETRAINING_LOG_PATH, REVIEWER_SCORES_PATH, RETRAINING_TRIGGERED_LOG_PATH

router = APIRouter()

def load_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]

def append_jsonl(path: Path, obj: dict):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")

def map_score_to_weight(score: float) -> float:
    if score >= 0.75:
        return 1.25
    elif score >= 0.5:
        return 1.0
    else:
        return 0.75

def get_consensus_status(signal_id: str) -> dict:
    retraining_entries = load_jsonl(RETRAINING_LOG_PATH)
    matching = [e for e in retraining_entries if e.get("signal_id") == signal_id]

    if not matching:
        raise HTTPException(status_code=404, detail="No entries for this signal_id")

    seen: Dict[str, dict] = {}
    for entry in matching:
        rid = entry["reviewer_id"]
        if rid not in seen:
            seen[rid] = entry  # only first flag counts

    scores_by_id = {}
    if REVIEWER_SCORES_PATH.exists():
        for rec in load_jsonl(REVIEWER_SCORES_PATH):
            scores_by_id[rec["reviewer_id"]] = rec.get("score")

    reviewers: List[Dict[str, float]] = []
    combined_weight = 0.0

    for rid, entry in seen.items():
        if "reviewer_weight" in entry:
            weight = entry["reviewer_weight"]
        else:
            score = scores_by_id.get(rid)
            weight = map_score_to_weight(score) if score is not None else 1.0

        reviewers.append({"reviewer_id": rid, "weight": round(weight, 2)})
        combined_weight += weight

    return {
        "signal_id": signal_id,
        "total_reviewers": len(reviewers),
        "combined_weight": round(combined_weight, 2),
        "reviewers": reviewers
    }

@router.get("/consensus-status/{signal_id}")
def get_status(signal_id: str):
    return get_consensus_status(signal_id)

@router.post("/evaluate-consensus-retraining")
def evaluate_consensus_retraining(
    payload: dict = Body(...),
    threshold: float = 2.5
):
    signal_id = payload.get("signal_id")
    if not signal_id:
        raise HTTPException(status_code=400, detail="Missing signal_id in request body")

    try:
        consensus = get_consensus_status(signal_id)
    except HTTPException as e:
        if e.status_code == 404:
            return {
                "triggered": False,
                "total_weight": 0.0,
                "reviewers": []
            }
        raise

    total = consensus["combined_weight"]

    if total >= threshold:
        trigger_entry = {
            "signal_id": signal_id,
            "timestamp": time.time(),
            "total_weight": total,
            "reviewers": consensus["reviewers"],
        }
        append_jsonl(RETRAINING_TRIGGERED_LOG_PATH, trigger_entry)
        return {
            "triggered": True,
            **consensus
        }

    return {
        "triggered": False,
        **consensus
    }
