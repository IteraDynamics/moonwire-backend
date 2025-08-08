# src/consensus_router.py

import json
from fastapi import APIRouter, HTTPException, Query
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

CONSENSUS_THRESHOLD = 2.5

router = APIRouter(prefix="/internal")


def _score_to_weight(score: Optional[float]) -> float:
    """Map raw reviewer score to consensus weight bands."""
    if score is None:
        return 1.0
    if score >= 0.75:
        return 1.25
    if score >= 0.5:
        return 1.0
    return 0.75


# ---------- DEBUG (raw score fallback) ----------
@router.get("/consensus-debug/{signal_id}")
def consensus_debug(signal_id: str):
    """
    Audit trail for a single signal. Fallback uses RAW score (no banding).
    """
    # Import paths inside the function so pytest's reload() is respected
    import src.paths as paths

    log_path = Path(paths.RETRAINING_LOG_PATH)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No retraining log found")

    # Load reviewer scores (raw; no banding in debug)
    scores_path = Path(paths.REVIEWER_SCORES_PATH)
    raw_scores: Dict[str, float] = {}
    if scores_path.exists():
        with scores_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                raw_scores[rec["reviewer_id"]] = rec.get("score", 1.0)

    all_flags = []
    seen_reviewers = set()
    counted_reviewers = []
    total_weight_used = 0.0

    with log_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("signal_id") != signal_id:
                continue

            reviewer_id = entry["reviewer_id"]
            weight = entry.get("reviewer_weight")
            if weight is None:
                weight = raw_scores.get(reviewer_id, 1.0)  # RAW for debug

            is_duplicate = reviewer_id in seen_reviewers
            all_flags.append({
                "reviewer_id": reviewer_id,
                "reviewer_weight": weight,
                "timestamp": datetime.fromtimestamp(entry["timestamp"]).isoformat(),
                "duplicate": is_duplicate
            })

            if not is_duplicate:
                seen_reviewers.add(reviewer_id)
                counted_reviewers.append(reviewer_id)
                total_weight_used += weight

    if not all_flags:
        raise HTTPException(status_code=404, detail="No entries for this signal_id")

    return {
        "signal_id": signal_id,
        "all_flags": all_flags,
        "counted_reviewers": counted_reviewers,
        "total_weight_used": total_weight_used,
        "threshold": CONSENSUS_THRESHOLD,
        "triggered": total_weight_used >= CONSENSUS_THRESHOLD
    }


# ---------- EVALUATE (banded fallback) ----------
@router.post("/evaluate-consensus-retraining")
def evaluate_consensus_retraining(payload: Dict[str, str]):
    """
    Threshold decision endpoint. Fallback uses BANDING (1.25/1.0/0.75).
    Also writes a trigger log line when threshold is met.
    """
    import src.paths as paths

    signal_id = payload.get("signal_id")
    if not signal_id:
        raise HTTPException(status_code=400, detail="Missing signal_id")

    log_path = Path(paths.RETRAINING_LOG_PATH)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Retraining log not found")

    # Load reviewer scores (raw), apply banding when used
    scores_path = Path(paths.REVIEWER_SCORES_PATH)
    raw_scores: Dict[str, float] = {}
    if scores_path.exists():
        with scores_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                raw_scores[rec["reviewer_id"]] = rec.get("score")

    seen = set()
    total_weight = 0.0
    reviewer_weights: Dict[str, float] = {}

    with log_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("signal_id") != signal_id:
                continue

            reviewer_id = entry["reviewer_id"]
            if reviewer_id in seen:
                continue

            weight = entry.get("reviewer_weight")
            if weight is None:
                weight = _score_to_weight(raw_scores.get(reviewer_id))

            seen.add(reviewer_id)
            total_weight += weight
            reviewer_weights[reviewer_id] = weight

    triggered = total_weight >= CONSENSUS_THRESHOLD

    # Write trigger log if threshold met
    if triggered:
        trig_path = Path(paths.RETRAINING_TRIGGERED_LOG_PATH)
        trig_path.parent.mkdir(parents=True, exist_ok=True)
        log_entry = {
            "signal_id": signal_id,
            "total_weight": total_weight,
            "threshold": CONSENSUS_THRESHOLD,
            "reviewers": [{"id": r, "weight": reviewer_weights[r]} for r in seen],
            "timestamp": datetime.utcnow().isoformat()
        }
        with trig_path.open("a") as f:
            f.write(json.dumps(log_entry) + "\n")

    return {
        "signal_id": signal_id,
        "total_weight": total_weight,
        "triggered": triggered,
        "threshold": CONSENSUS_THRESHOLD,
        "reviewers": [{"id": r, "weight": reviewer_weights[r]} for r in seen]
    }


# ---------- SIMULATE (banded fallback, read-only) ----------
@router.get("/consensus-simulate/{signal_id}")
def consensus_simulate(signal_id: str, threshold: Optional[float] = None):
    """
    Read-only replay: dedupe by reviewer, resolve weights using BANDING fallback,
    sum, and compare to provided threshold (or system default).
    """
    import src.paths as paths

    thr = threshold if threshold is not None else CONSENSUS_THRESHOLD

    log_path = Path(paths.RETRAINING_LOG_PATH)
    if not log_path.exists():
        # No log file: empty result, no trigger
        return {
            "signal_id": signal_id,
            "threshold_tested": thr,
            "total_weight": 0.0,
            "would_trigger": False,
            "counted_reviewers": []
        }

    # Load reviewer scores (raw), band when missing reviewer_weight
    scores_path = Path(paths.REVIEWER_SCORES_PATH)
    raw_scores: Dict[str, float] = {}
    if scores_path.exists():
        with scores_path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                raw_scores[rec["reviewer_id"]] = rec.get("score")

    seen = set()
    counted = []
    total_weight = 0.0

    with log_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get("signal_id") != signal_id:
                continue

            rid = entry["reviewer_id"]
            if rid in seen:
                continue
            seen.add(rid)

            weight = entry.get("reviewer_weight")
            if weight is None:
                weight = _score_to_weight(raw_scores.get(rid))

            counted.append({"reviewer_id": rid, "weight": weight})
            total_weight += weight

    return {
        "signal_id": signal_id,
        "threshold_tested": thr,
        "total_weight": total_weight,
        "would_trigger": total_weight >= thr,
        "counted_reviewers": counted
    }


# ---------- REVIEWER LEADERBOARD ----------
@router.get("/reviewer-leaderboard")
def reviewer_leaderboard(limit: int = Query(10, ge=1, le=100)):
    """
    Return the top reviewers by trust-weight (banded from score).
    - Reads reviewer_scores.jsonl
    - Dedupes by reviewer_id using the most recent entry (last occurrence wins)
    - Sorts high→low by weight, then score
    - last_updated is ISO8601 if timestamp/updated_at present; otherwise null
    """
    import src.paths as paths

    scores_path = Path(paths.REVIEWER_SCORES_PATH)
    if not scores_path.exists():
        return {"leaderboard": []}

    latest: Dict[str, Dict] = {}   # reviewer_id -> {score, last_updated}
    order: List[str] = []          # to preserve "last write wins" if no timestamp

    with scores_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            rid = rec.get("reviewer_id")
            if not rid:
                continue

            score = rec.get("score", None)
            ts = rec.get("timestamp", rec.get("updated_at"))
            ts_iso: Optional[str] = None
            if isinstance(ts, (int, float)):
                ts_iso = datetime.utcfromtimestamp(ts).isoformat() + "Z"
            elif isinstance(ts, str):
                ts_iso = ts

            if rid not in latest:
                order.append(rid)
                latest[rid] = {"score": score, "last_updated": ts_iso}
            else:
                prev_ts = latest[rid].get("last_updated")
                if ts_iso and prev_ts:
                    if ts_iso >= prev_ts:
                        latest[rid] = {"score": score, "last_updated": ts_iso}
                else:
                    latest[rid] = {"score": score, "last_updated": ts_iso}

    rows = []
    for rid in order:
        if rid not in latest:
            continue
        score = latest[rid].get("score")
        weight = _score_to_weight(score)
        rows.append({
            "reviewer_id": rid,
            "score": score,
            "weight": weight,
            "last_updated": latest[rid].get("last_updated")
        })

    # Sort: weight desc, then score desc (None goes last)
    def sort_key(row):
        score = row["score"]
        score_sort = score if isinstance(score, (int, float)) else -1
        return (-row["weight"], -score_sort)

    rows.sort(key=sort_key)
    return {"leaderboard": rows[:limit]}