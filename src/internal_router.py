from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from collections import defaultdict
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import os
import logging
import requests
from src.signal_utils import compute_trust_scores
from src.reviewer_impact_logger import log_reviewer_impact
from src.reviewer_impact_logger import ReviewerImpactLog
from src.reviewer_log_utils import log_reviewer_action
from src.jsonl_writer import atomic_jsonl_append
from src.paths import (
    FEEDBACK_LOG_PATH,
    SUPPRESSION_REVIEW_PATH,
    RETRAIN_QUEUE_PATH
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/internal", tags=["internal-tools"])

# === Feedback Summary Route ===
@router.get("/feedback-summary")
def get_feedback_summary():
    if not FEEDBACK_LOG_PATH.exists():
        return {
            "total_feedback": 0,
            "agree_percentage": 0.0,
            "disagree_count": 0,
            "most_disagreed_signals": []
        }

    total = 0
    agree = 0
    disagree_signals = defaultdict(list)

    with FEEDBACK_LOG_PATH.open("r") as f:
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

# === Update Suppression Status (Task 2 + Task 1 impact score) ===
class SuppressionStatusUpdate(BaseModel):
    signal_id: str
    status: str  # "reviewed", "ignored", "retrained", "overridden"
    reviewer_id: str
    note: Optional[str] = None

@router.post("/update-suppression-status")
def update_suppression_status(update: SuppressionStatusUpdate):
    if not SUPPRESSION_REVIEW_PATH.exists():
        raise HTTPException(status_code=404, detail="Suppression log not found.")

    valid_statuses = {"reviewed", "ignored", "retrained", "overridden"}
    if update.status not in valid_statuses:
        raise HTTPException(status_code=400, detail="Invalid status value.")

    action_weights = {"ignored": 0.1, "reviewed": 0.3, "retrained": 0.6, "overridden": 1.0}
    now = datetime.now(timezone.utc)
    updated_entries = []
    found = False
    status_changed = False

    all_entries = []
    with SUPPRESSION_REVIEW_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                all_entries.append(entry)
            except json.JSONDecodeError:
                continue

    for entry in all_entries:
        if entry.get("id") == update.signal_id:
            found = True
            if entry.get("status") != update.status:
                entry["status"] = update.status
                entry["reviewer_id"] = update.reviewer_id
                entry["note"] = update.note or ""
                entry["last_updated"] = now.isoformat()
                status_changed = True

                trust_score = entry.get("trust_score", 0.5)
                action_weight = action_weights[update.status]
                recurrence_bonus = 0.0

                recent_count = sum(
                    1 for e in all_entries
                    if e.get("asset") == entry.get("asset")
                    and e.get("id") != entry.get("id")
                    and e.get("timestamp")
                    and datetime.fromisoformat(e["timestamp"]) >= now - timedelta(hours=24)
                )
                if recent_count >= 2:
                    recurrence_bonus = 0.1

                entry["impact_score"] = round(action_weight * (1 - trust_score) + recurrence_bonus, 3)

                # ✅ Log reviewer impact (this is the key Step 2 part)
                try:
                    log_reviewer_impact(
                        reviewer_id=update.reviewer_id,
                        signal_id=update.signal_id,
                        action_type=update.status,
                        original_trust_score=trust_score,
                        trust_score_before=trust_score,
                        trust_score_after=trust_score if update.status == "overridden" else None,
                        signal_timestamp=entry.get("timestamp"),
                        reviewer_note=update.note,
                        reason=entry.get("retrain_hint", "unspecified"),
                        model_version=entry.get("model_version", "v1.0")
                    )
                except Exception as e:
                    print(f"❌ Reviewer impact log failed: {e}")

        updated_entries.append(entry)

    if not found:
        raise HTTPException(status_code=404, detail="Signal ID not found.")

    if not status_changed:
        return {"updated": False, "reason": "Status already up to date."}

    with SUPPRESSION_REVIEW_PATH.open("w") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")

    return {
        "updated": True,
        "signal_id": update.signal_id,
        "new_status": update.status
    }

# === Review Status Summary (extended for impact score) ===
@router.get("/review-status-summary")
def review_status_summary():
    if not SUPPRESSION_REVIEW_PATH.exists():
        return {
            "total_reviewed": 0,
            "status_counts": {},
            "hint_breakdown": {},
            "average_impact_score_by_status": {},
            "top_5_most_impactful": []
        }

    status_counts = defaultdict(int)
    hint_breakdown = defaultdict(int)
    impact_totals = defaultdict(float)
    impact_counts = defaultdict(int)
    impactful = []
    total_reviewed = 0

    with SUPPRESSION_REVIEW_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                status = entry.get("status")
                if status and status != "pending":
                    total_reviewed += 1
                    status_counts[status] += 1
                    hint = entry.get("retrain_hint")
                    if hint:
                        hint_breakdown[hint] += 1
                    if "impact_score" in entry:
                        impact_totals[status] += entry["impact_score"]
                        impact_counts[status] += 1
                        impactful.append(entry)
            except json.JSONDecodeError:
                continue

    avg_impact = {
        status: round(impact_totals[status] / impact_counts[status], 3)
        for status in impact_totals
        if impact_counts[status] > 0
    }

    top_impactful = sorted(impactful, key=lambda x: x.get("impact_score", 0), reverse=True)[:5]

    return {
        "total_reviewed": total_reviewed,
        "status_counts": dict(status_counts),
        "hint_breakdown": dict(hint_breakdown),
        "average_impact_score_by_status": avg_impact,
        "top_5_most_impactful": top_impactful
    }

# === Suppression Pattern Intelligence (Task 1 – 7/10) ===
@router.get("/suppression-patterns")
def suppression_pattern_summary(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    trust_band: Optional[str] = Query(None)
):
    if not SUPPRESSION_REVIEW_PATH.exists():
        return {"patterns": []}

    pattern_groups = defaultdict(list)
    now = datetime.now(timezone.utc)

    start_dt = datetime.fromisoformat(start_date) if start_date else datetime.min
    end_dt = datetime.fromisoformat(end_date) if end_date else now

    with SUPPRESSION_REVIEW_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)

                if entry.get("status") == "pending":
                    continue

                ts = entry.get("timestamp")
                if not ts:
                    continue

                ts_dt = datetime.fromisoformat(ts)
                if not (start_dt <= ts_dt <= end_dt):
                    continue

                if trust_band and entry.get("trust_label") != trust_band:
                    continue

                key = (
                    entry.get("label", "unknown"),
                    entry.get("fallback_type", "unknown"),
                    entry.get("retrain_hint", "none")
                )
                pattern_groups[key].append(entry)

            except Exception:
                continue

    output = []
    for (label, fallback, hint), group in pattern_groups.items():
        trust_scores = [e.get("trust_score", 0.5) for e in group]
        impact_scores = [e.get("impact_score", 0.0) for e in group]
        overridden_count = sum(1 for e in group if e.get("status") == "overridden")

        avg_trust = round(sum(trust_scores) / len(trust_scores), 3)
        avg_impact = round(sum(impact_scores) / len(impact_scores), 3)
        avg_priority = round((1 - avg_trust) * avg_impact, 3)

        output.append({
            "label_type": label,
            "fallback_type": fallback,
            "retrain_hint": hint,
            "count": len(group),
            "avg_trust_score": avg_trust,
            "avg_impact_score": avg_impact,
            "avg_priority_score": avg_priority,
            "overridden_count": overridden_count
        })

    output_sorted = sorted(output, key=lambda x: (-x["count"], -x["avg_priority_score"]))

    return {"patterns": output_sorted[:10]}

# === Trust Breakdown Timeline (Task 2 – 7/10) ===
@router.get("/trust-breakdown-timeline")
def trust_breakdown_timeline(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    asset: Optional[str] = Query(None),
    label_type: Optional[str] = Query(None)
):
    if not SUPPRESSION_REVIEW_PATH.exists():
        return {"timeline": []}

    timeline_map = defaultdict(list)
    start_dt = datetime.fromisoformat(start_date) if start_date else datetime.min
    end_dt = datetime.fromisoformat(end_date) if end_date else datetime.now(timezone.utc)

    with SUPPRESSION_REVIEW_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                ts = entry.get("timestamp")
                if not ts:
                    continue
                ts_dt = datetime.fromisoformat(ts)
                if not (start_dt <= ts_dt <= end_dt):
                    continue

                if asset and entry.get("asset") != asset:
                    continue
                if label_type and entry.get("label") != label_type:
                    continue

                date_key = ts_dt.date().isoformat()
                timeline_map[date_key].append(entry)
            except Exception:
                continue

    timeline = []
    for date_str, entries in sorted(timeline_map.items()):
        total = len(entries)
        suppressed = sum(1 for e in entries if e.get("status") in {"reviewed", "ignored", "retrained", "overridden"})
        overridden = sum(1 for e in entries if e.get("status") == "overridden")
        retrain_flags = sum(1 for e in entries if e.get("status") == "retrained")

        trust_scores = [e.get("trust_score", 0.5) for e in entries]
        avg_trust = round(sum(trust_scores) / len(trust_scores), 3) if trust_scores else 0.5

        label_counts = defaultdict(int)
        asset_counts = defaultdict(int)
        for e in entries:
            label_counts[e.get("label", "unknown")] += 1
            asset_counts[e.get("asset", "unknown")] += 1

        top_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:3]
        top_assets = sorted(asset_counts.items(), key=lambda x: -x[1])[:3]

        timeline.append({
            "date": date_str,
            "total_signals": total,
            "suppressed_count": suppressed,
            "avg_trust_score": avg_trust,
            "overridden_count": overridden,
            "retrain_flag_count": retrain_flags,
            "top_labels_triggered": top_labels,
            "top_assets_triggered": top_assets
        })

    return {"timeline": timeline}
    
@router.get("/trust-today")
def trust_today_diagnostics():
    if not SUPPRESSION_REVIEW_PATH.exists():
        return {"error": "Suppression log not found."}

    today = datetime.now(timezone.utc).date()
    total_signals = 0
    suppressed = 0
    overridden = 0
    retrain_flags = 0
    trust_scores = []
    retrain_hints = defaultdict(int)
    asset_priority = defaultdict(float)

    with SUPPRESSION_REVIEW_PATH.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                ts = entry.get("timestamp")
                if not ts:
                    continue
                ts_dt = datetime.fromisoformat(ts)
                if ts_dt.date() != today:
                    continue

                total_signals += 1
                status = entry.get("status")
                if status in {"reviewed", "ignored", "retrained", "overridden"}:
                    suppressed += 1
                if status == "overridden":
                    overridden += 1
                if status == "retrained":
                    retrain_flags += 1

                trust_scores.append(entry.get("trust_score", 0.5))
                hint = entry.get("retrain_hint")
                if hint:
                    retrain_hints[hint] += 1

                asset = entry.get("asset", "unknown")
                priority = entry.get("impact_score", 0.0)
                asset_priority[asset] += priority
            except Exception:
                continue

    avg_trust = round(sum(trust_scores) / len(trust_scores), 3) if trust_scores else 0.5
    top_assets = sorted(asset_priority.items(), key=lambda x: -x[1])[:5]
    top_hints = sorted(retrain_hints.items(), key=lambda x: -x[1])[:5]

    return {
        "total_signals_today": total_signals,
        "suppressed_today": suppressed,
        "overrides_today": overridden,
        "retrain_flags_today": retrain_flags,
        "avg_trust_score_today": avg_trust,
        "top_risky_assets": top_assets,
        "most_common_retrain_hints": top_hints,
        "last_updated_at": datetime.now(timezone.utc).isoformat()
    }

class SuppressedSignal(BaseModel):
    signal_id: str
    asset: str
    trust_score: float
    suppression_reason: str
    label: Optional[str] = None
    score: Optional[float] = None
    confidence: Optional[float] = None
    likely_disagreed: Optional[bool] = None
    retrain_hint: Optional[str] = None
    model_version: Optional[str] = "unknown"

@router.post("/log-signal-for-review")
def log_signal_for_review(signal: SuppressedSignal):
    if not signal.suppression_reason:
        raise HTTPException(status_code=400, detail="suppression_reason is required.")

    entry = {
        "id": signal.signal_id,
        "asset": signal.asset,
        "trust_score": signal.trust_score,
        "suppression_reason": signal.suppression_reason,
        "status": "pending",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": signal.label,
        "score": signal.score,
        "confidence": signal.confidence,
        "likely_disagreed": signal.likely_disagreed,
        "retrain_hint": signal.retrain_hint,
        "model_version": signal.model_version,
    }

    try:
        atomic_jsonl_append(SUPPRESSION_REVIEW_PATH, entry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write suppression log: {e}")

    return {"logged": True, "signal_id": signal.signal_id}

