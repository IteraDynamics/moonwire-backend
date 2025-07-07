from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from collections import defaultdict
from typing import List
from datetime import datetime
import json
import os
import requests
from pathlib import Path
from src.signal_utils import compute_trust_scores

router = APIRouter(prefix="/internal", tags=["internal-tools"])

FEEDBACK_LOG_PATH = "data/feedback.jsonl"
SUPPRESSION_REVIEW_PATH = Path("data/suppression_review_queue.jsonl")
RETRAIN_QUEUE_PATH = Path("data/retrain_queue.jsonl")

# === Feedback Summary Route ===
@router.get("/feedback-summary")
def get_feedback_summary():
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return {
            "total_feedback": 0,
            "agree_percentage": 0.0,
            "disagree_count": 0,
            "most_disagreed_signals": []
        }

    total = 0
    agree = 0
    disagree_signals = defaultdict(list)

    with open(FEEDBACK_LOG_PATH, "r") as f:
        for line in f:
            try:
                fb = json.loads(line)
                total += 1
                if fb.get("agree") is True:
                    agree += 1
                else:
                    sid = fb.get("signal_id", "unknown")
                    disagree_signals[sid].append(fb.get("note", ""))
            except json.JSONDecodeError:
                continue

    top_signals = sorted(disagree_signals.items(), key=lambda x: len(x[1]), reverse=True)[:3]
    most_disagreed = [
        {
            "signal_id": sid,
            "count": len(notes),
            "notes": [n for n in notes if n]
        }
        for sid, notes in top_signals
    ]

    return {
        "total_feedback": total,
        "agree_percentage": round((agree / total) * 100, 2) if total > 0 else 0.0,
        "disagree_count": total - agree,
        "most_disagreed_signals": most_disagreed
    }

# === Signal Trust Score Route ===
@router.get("/signal-trust-insights")
def signal_trust_insights():
    def fetch_disagreement_prediction(payload):
        response = requests.post(
            "http://localhost:8000/internal/predict-feedback-risk",
            json=payload
        )
        if response.status_code == 200:
            return response.json()
        return {"probability": 0.5}

    return compute_trust_scores(fetch_disagreement_prediction)

# === Suppression Review Queue Route ===
@router.get("/review-suppressed")
def review_suppressed_signals():
    if not SUPPRESSION_REVIEW_PATH.exists():
        return {"review_queue": []}

    pending_signals = []
    with SUPPRESSION_REVIEW_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("status") == "pending":
                    pending_signals.append(entry)
            except json.JSONDecodeError:
                continue

    return {"review_queue": pending_signals}

# === Suppression Status Update ===
class SuppressionUpdate(BaseModel):
    signal_id: str
    new_status: str  # "reviewed" or "flag_for_retraining"

@router.post("/mark-suppressed")
def mark_suppressed(update: SuppressionUpdate):
    if update.new_status not in ["reviewed", "flag_for_retraining"]:
        raise HTTPException(status_code=400, detail="Invalid status")

    if not SUPPRESSION_REVIEW_PATH.exists():
        raise HTTPException(status_code=404, detail="No suppression file found")

    updated_entries = []
    updated = False

    with SUPPRESSION_REVIEW_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("id") == update.signal_id and entry.get("status") == "pending":
                    entry["status"] = update.new_status
                    updated = True
                updated_entries.append(entry)
            except json.JSONDecodeError:
                continue

    if not updated:
        raise HTTPException(status_code=404, detail="Pending signal not found")

    with SUPPRESSION_REVIEW_PATH.open("w") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")

    return {"updated": True, "signal_id": update.signal_id, "new_status": update.new_status}

# === Batch Flag Suppressed Signals for Retraining ===
class BatchFlagFilters(BaseModel):
    retrain_hint: str | None = None
    asset: str | None = None
    trust_score_below: float | None = None

class BatchFlagRequest(BaseModel):
    filters: BatchFlagFilters
    note: str | None = None

@router.post("/batch-flag-retraining")
def batch_flag_retraining(payload: BatchFlagRequest, dry_run: bool = Query(False)):
    filters = payload.filters
    note = payload.note or ""

    batch_flagged = 0
    skipped_duplicates = 0
    flagged_ids = set()

    retrain_existing_ids = set()
    if RETRAIN_QUEUE_PATH.exists():
        with RETRAIN_QUEUE_PATH.open("r") as f:
            for line in f:
                try:
                    retrain_existing_ids.add(json.loads(line)["id"])
                except:
                    continue

    to_append = []

    if SUPPRESSION_REVIEW_PATH.exists():
        with SUPPRESSION_REVIEW_PATH.open("r") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except:
                    continue

                if entry.get("status") != "pending":
                    continue

                if filters.retrain_hint and entry.get("retrain_hint") != filters.retrain_hint:
                    continue
                if filters.asset and entry.get("asset") != filters.asset:
                    continue
                if filters.trust_score_below is not None and entry.get("trust_score", 1.0) >= filters.trust_score_below:
                    continue

                if entry["id"] in retrain_existing_ids or entry["id"] in flagged_ids:
                    skipped_duplicates += 1
                    continue

                flagged_ids.add(entry["id"])

                enriched = {
                    **entry,
                    "flagged_for_retraining": True,
                    "flag_reason": "batch_filter",
                    "flag_source": "batch_route",
                    "flagged_at": datetime.utcnow().isoformat(),
                    "note": note
                }

                to_append.append(enriched)

    if not dry_run and to_append:
        os.makedirs(RETRAIN_QUEUE_PATH.parent, exist_ok=True)
        with RETRAIN_QUEUE_PATH.open("a") as f:
            for item in to_append:
                f.write(json.dumps(item) + "\n")

    return {
        "batch_flagged": len(to_append),
        "skipped_duplicates": skipped_duplicates,
        "filters_applied": filters.dict(),
        "dry_run": dry_run
    }