from __future__ import annotations
import os, random
from fastapi import APIRouter, HTTPException, Query
from src.analytics.lead_lag import compute_lead_lag
from src.paths import LOGS_DIR

router = APIRouter()  # prefix added when included in main.py

def _demo_seed(days: int, interval: str, max_lag: int, use: str):
    origins = ["twitter", "reddit", "rss_news"]
    seeded = []
    for i in range(len(origins)):
        for j in range(len(origins)):
            if i == j:
                continue
            L = random.randint(1, max(1, max_lag))
            r = round(random.uniform(0.3, 0.8), 3)
            a, b = origins[i], origins[j]
            seeded.append({
                "a": a,
                "b": b,
                "best_lag": L,           # positive => a leads b
                "correlation": r,
                "leader": a
            })
    seeded.sort(key=lambda p: (-abs(p["correlation"]), p["a"], p["b"]))
    return {
        "window_days": days,
        "interval": interval,
        "max_lag": max_lag,
        "use": use,
        "origins": origins,
        "pairs": seeded,
        "notes": ["demo seeded"]
    }

@router.get("/lead-lag")
def lead_lag(
    days: int = Query(7, ge=1),
    interval: str = Query("hour"),
    max_lag: int = Query(24, ge=0),
    top: int = Query(5, ge=1),
    use: str = Query("flags")
):
    if interval not in ("hour", "day"):
        raise HTTPException(status_code=422, detail='interval must be "hour" or "day"')
    if use not in ("flags", "triggers"):
        raise HTTPException(status_code=422, detail='use must be "flags" or "triggers"')

    data = compute_lead_lag(
        LOGS_DIR / "retraining_log.jsonl",
        LOGS_DIR / "retraining_triggered.jsonl",
        days=days,
        interval=interval,
        max_lag=max_lag,
        use=use
    )

    if (not data["pairs"]) and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        data = _demo_seed(days, interval, max_lag, use)

    # Return only top N pairs
    data["pairs"] = data["pairs"][:top]
    return data
