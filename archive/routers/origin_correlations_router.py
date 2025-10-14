from __future__ import annotations
import os, random
from fastapi import APIRouter, HTTPException, Query
from src.analytics.origin_correlations import compute_origin_correlations
from src.paths import LOGS_DIR

router = APIRouter(prefix="/internal", tags=["internal"])

def _demo_seed(days: int, interval: str):
    origins = ["reddit", "rss_news", "twitter"]
    seeded = []
    for i in range(len(origins)):
        for j in range(i + 1, len(origins)):
            seeded.append({
                "a": origins[i],
                "b": origins[j],
                "correlation": round(random.uniform(0.3, 0.8), 3)
            })
    seeded.sort(key=lambda p: p["correlation"], reverse=True)
    return {"window_days": days, "interval": interval, "origins": sorted(origins), "pairs": seeded, "notes": ["demo seeded"]}

@router.get("/origin-correlations")
def origin_correlations(days: int = Query(7, ge=1), interval: str = Query("day")):
    if interval not in ("day", "hour"):
        raise HTTPException(status_code=400, detail='interval must be "day" or "hour"')
    data = compute_origin_correlations(
        LOGS_DIR / "retraining_log.jsonl",
        LOGS_DIR / "retraining_triggered.jsonl",
        days=days,
        interval=interval,
    )
    if (not data["pairs"]) and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        data = _demo_seed(days, interval)
    return data
