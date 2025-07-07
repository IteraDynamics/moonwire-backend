# src/internal_router.py

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from collections import defaultdict
from typing import List, Optional
from fastapi.responses import JSONResponse, StreamingResponse
import json
import os
import requests
from datetime import datetime
from src.signal_utils import compute_trust_scores

router = APIRouter(prefix="/internal", tags=["internal-tools"])

FEEDBACK_LOG_PATH = "data/feedback.jsonl"
SUPPRESSION_REVIEW_PATH = "data/suppression_review_queue.jsonl"
RETRAIN_QUEUE_PATH = "data/retrain_queue.jsonl"

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
    if not os.path.exists(SUPPRESSION_REVIEW_PATH):
        return {"review_queue": []}

    pending_signals = []
    with open(SUPPRESSION_REVIEW_PATH, "r") as f:
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

    if not os.path.exists(SUPPRESSION_REVIEW_PATH):
        raise HTTPException(status_code=404, detail="No suppression file found")

    updated_entries = []
    updated = False

    with open(SUPPRESSION_REVIEW_PATH, "r") as f:
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

    with open(SUPPRESSION_REVIEW_PATH, "w") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")

    return {"updated": True, "signal_id": update.signal_id, "new_status": update.new_status}

# === Export Retrain Queue ===
@router.get("/export-retrain-queue")
def export_retrain_queue(
    asset: Optional[str] = Query(None),
    hint: Optional[str] = Query(None),
    limit: Optional[int] = Query(None),
    as_file: Optional[bool] = Query(False)
):
    if not os.path.exists(RETRAIN_QUEUE_PATH):
        return {"exported": 0, "signals": []}

    signals_by_id = {}
    with open(RETRAIN_QUEUE_PATH, "r") as f:
        for line in f:
            try:
                signal = json.loads(line)
                signal_id = signal.get("id") or signal.get("signal_id")
                if not signal_id:
                    continue
                if asset and signal.get("asset") != asset:
                    continue
                if hint and signal.get("retrain_hint") != hint:
                    continue
                signals_by_id[signal_id] = signal
            except json.JSONDecodeError:
                continue

    signal_list = list(signals_by_id.values())
    if limit:
        signal_list = signal_list[:limit]

    if as_file:
        def iter_jsonl():
            for s in signal_list:
                yield json.dumps(s) + "\n"

        filename = f"retrain_export_{datetime.utcnow().isoformat()}.jsonl"
        return StreamingResponse(
            iter_jsonl(),
            media_type="application/jsonl",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    return {
        "exported": len(signal_list),
        "signals": signal_list
    }