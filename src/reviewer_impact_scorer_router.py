# src/reviewer_impact_scorer_router.py

from fastapi import APIRouter, Request
import json
import os

from src.paths import REVIEWER_IMPACT_LOG_PATH, REVIEWER_SCORES_PATH

router = APIRouter()

@router.post("/reviewer-impact-log")
async def log_reviewer_action(request: Request):
    # 1) Read the incoming JSON
    payload = await request.json()

    # 2) DEBUG: confirm what FastAPI parsed
    print("🧐 Payload received:", payload)

    # 3) Existing logging logic
    print("🚨 /internal/reviewer-impact-log hit")
    print(f"📄 Writing to: {REVIEWER_IMPACT_LOG_PATH}")
    try:
        os.makedirs(REVIEWER_IMPACT_LOG_PATH.parent, exist_ok=True)
        with REVIEWER_IMPACT_LOG_PATH.open("a") as f:
            f.write(json.dumps(payload) + "\n")
        print("✅ Log entry appended.")
        return {"status": "logged"}
    except Exception as e:
        print(f"❌ Failed to write log: {e}")
        return {"error": str(e)}

@router.post("/trigger-reviewer-scoring")
async def trigger_scoring():
    print("🚨 /internal/trigger-reviewer-scoring hit")
    if not REVIEWER_IMPACT_LOG_PATH.exists() or REVIEWER_IMPACT_LOG_PATH.stat().st_size == 0:
        print("⚠️ No reviewer logs to score.")
        return {"error": "No reviewer logs to score."}
    try:
        with REVIEWER_IMPACT_LOG_PATH.open("r") as f:
            logs = [json.loads(line) for line in f if line.strip()]
        print(f"📊 Loaded {len(logs)} log entries")

        reviewer_scores = {}
        for log in logs:
            reviewer_id = log.get("reviewer_id")
            if reviewer_id:
                reviewer_scores[reviewer_id] = reviewer_scores.get(reviewer_id, 0) + 1

        os.makedirs(REVIEWER_SCORES_PATH.parent, exist_ok=True)
        with REVIEWER_SCORES_PATH.open("w") as f:
            for reviewer_id, score in reviewer_scores.items():
                f.write(json.dumps({"reviewer_id": reviewer_id, "score": score}) + "\n")

        print(f"✅ {len(reviewer_scores)} reviewer scores written to {REVIEWER_SCORES_PATH}")
        return {"recomputed": True}
    except Exception as e:
        print(f"❌ Scoring failed: {e}")
        return {"error": str(e)}

@router.get("/reviewer-scores")
async def get_reviewer_scores():
    print("📥 /internal/reviewer-scores requested")
    if not REVIEWER_SCORES_PATH.exists():
        print("⚠️ No reviewer_scores.jsonl found.")
        return {"scores": []}
    try:
        with REVIEWER_SCORES_PATH.open("r") as f:
            scores = [json.loads(line) for line in f if line.strip()]
        print(f"📊 Returning {len(scores)} scores")
        return {"scores": scores}
    except Exception as e:
        print(f"❌ Failed to read scores: {e}")
        return {"error": str(e)}

@router.get("/debug/jsonl-status")
async def jsonl_status():
    print("🧪 /internal/debug/jsonl-status hit")
    files = {
        "reviewer_impact_log": REVIEWER_IMPACT_LOG_PATH,
        "reviewer_scores": REVIEWER_SCORES_PATH,
    }
    status = {}
    for label, path in files.items():
        abs_path = str(path.resolve())
        exists = path.exists()
        writable = os.access(path, os.W_OK) if exists else False
        size = path.stat().st_size if exists else 0
        status[label] = {
            "exists": exists,
            "size_bytes": size,
            "writable": writable,
            "absolute_path": abs_path,
        }
        print(f"🔍 {label} — exists: {exists}, writable: {writable}, size: {size} bytes")
    return status
