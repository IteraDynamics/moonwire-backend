# src/reviewer_impact_scorer_router.py

from fastapi import APIRouter, Request
import json
import os

from src.paths import REVIEWER_IMPACT_LOG_PATH, REVIEWER_SCORES_PATH

# 🚀 Loaded reviewer_impact_scorer_router with trust-weighted logging
print("🚀 Loaded reviewer_impact_scorer_router v2 with trust-weighted logging")

router = APIRouter()

@router.post("/reviewer-impact-log")
async def log_reviewer_action(request: Request):
    # 1) Read the incoming JSON
    payload = await request.json()

    # DEBUG: confirm what FastAPI parsed
    print("🧐 Payload received:", payload)
    print("🚨 /internal/reviewer-impact-log hit")

    # 2) Load existing reviewer scores
    scores = {}
    if REVIEWER_SCORES_PATH.exists():
        try:
            with REVIEWER_SCORES_PATH.open("r") as sf:
                for line in sf:
                    if line.strip():
                        entry = json.loads(line)
                        rid = entry.get("reviewer_id")
                        scores[rid] = entry.get("score", 0)
        except Exception as e:
            print(f"❌ Failed to load reviewer scores: {e}")

    # 3) Determine weight multiplier based on score
    reviewer_id = payload.get("reviewer_id")
    raw_score = scores.get(reviewer_id, 0)
    if raw_score >= 0.75:
        weight = 1.25
    elif raw_score >= 0.5:
        weight = 1.0
    else:
        weight = 0.75

    # Attach weight to payload
    payload["reviewer_weight"] = weight
    print(f"⚖️ Reviewer {reviewer_id} weight: {weight} (score: {raw_score})")

    # 4) Append to impact log
    print(f"📄 Writing to: {REVIEWER_IMPACT_LOG_PATH}")
    try:
        os.makedirs(REVIEWER_IMPACT_LOG_PATH.parent, exist_ok=True)
        with REVIEWER_IMPACT_LOG_PATH.open("a") as f:
            f.write(json.dumps(payload) + "\n")
        print("✅ Log entry appended.")
        return {"status": "logged", "reviewer_weight": weight}
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
            rid = log.get("reviewer_id")
            if rid:
                reviewer_scores[rid] = reviewer_scores.get(rid, 0) + 1

        os.makedirs(REVIEWER_SCORES_PATH.parent, exist_ok=True)
        with REVIEWER_SCORES_PATH.open("w") as f:
            for rid, score in reviewer_scores.items():
                f.write(json.dumps({"reviewer_id": rid, "score": score}) + "\n")

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
