from fastapi import APIRouter, Request
from pathlib import Path
import json
import os

router = APIRouter()

LOG_PATH = Path("logs/reviewer_impact_log.jsonl")
SCORES_PATH = Path("logs/reviewer_scores.jsonl")

@router.post("/reviewer-impact-log")
async def log_reviewer_action(request: Request):
    payload = await request.json()
    print("🚨 /internal/reviewer-impact-log hit")
    print(f"📄 Writing to: {LOG_PATH}")
    try:
        os.makedirs(LOG_PATH.parent, exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(payload) + "\n")
        print("✅ Log written successfully.")
        return {"status": "logged"}
    except Exception as e:
        print(f"❌ Write failed: {e}")
        return {"error": str(e)}

@router.post("/trigger-reviewer-scoring")
async def trigger_scoring():
    print("🚨 /internal/trigger-reviewer-scoring hit")
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        print("⚠️ Log file missing or empty.")
        return {"error": "No reviewer logs to score."}

    try:
        with LOG_PATH.open("r") as f:
            logs = [json.loads(line) for line in f if line.strip()]

        reviewer_scores = {}
        for log in logs:
            reviewer_id = log.get("reviewer_id")
            if reviewer_id:
                reviewer_scores[reviewer_id] = reviewer_scores.get(reviewer_id, 0) + 1

        os.makedirs(SCORES_PATH.parent, exist_ok=True)
        with SCORES_PATH.open("w") as f:
            for reviewer_id, score in reviewer_scores.items():
                f.write(json.dumps({"reviewer_id": reviewer_id, "score": score}) + "\n")

        print(f"✅ Reviewer scores written to {SCORES_PATH}")
        return {"recomputed": True}
    except Exception as e:
        print(f"❌ Scoring failed: {e}")
        return {"error": str(e)}

@router.get("/reviewer-scores")
async def get_reviewer_scores():
    print("📥 /internal/reviewer-scores requested")
    if not SCORES_PATH.exists():
        print("⚠️ No reviewer_scores.jsonl found.")
        return {"scores": []}
    try:
        with SCORES_PATH.open("r") as f:
            scores = [json.loads(line) for line in f if line.strip()]
        return {"scores": scores}
    except Exception as e:
        print(f"❌ Failed to read scores: {e}")
        return {"error": str(e)}