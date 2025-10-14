from __future__ import annotations
import os, random
from fastapi import APIRouter, HTTPException, Query
from src.analytics.nowcast_attention import compute_nowcast_attention
from src.paths import LOGS_DIR

router = APIRouter(prefix="/internal", tags=["internal"])

def _demo_seed(days: int, interval: str, top: int):
    origins = ["twitter", "reddit", "rss_news"]
    seeded = []
    for o in origins:
        seeded.append({
            "origin": o,
            "score": round(random.uniform(55, 95), 1),
            "rank": 0,  # will be overwritten
            "components": {
                "z": round(random.uniform(1.0, 4.0), 2),
                "z_norm": round(random.uniform(0.2, 0.9), 2),
                "precision": round(random.uniform(0.4, 0.9), 2),
                "leadership": round(random.uniform(0.2, 0.8), 2),
                "regime": random.choice(["calm", "normal", "turbulent"]),
                "regime_factor": random.choice([0.95, 1.0, 1.05]),
                "threshold": random.choice([2.2, 2.5, 3.0]),
                "mean": round(random.uniform(0.3, 2.0), 2),
                "std": round(random.uniform(0.2, 1.5), 2),
                "current": random.randint(0, 8),
            }
        })
    seeded.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(seeded, 1):
        r["rank"] = i
    return {
        "window_days": days, "interval": interval,
        "as_of": None, "origins": seeded[:top],
        "notes": ["demo nowcast seeded"]
    }

@router.get("/nowcast-attention")
def nowcast_attention(
    days: int = Query(7, ge=1),
    interval: str = Query("hour"),
    lookback: int = Query(72, ge=2),
    z_cap: float = Query(5.0, gt=0),
    top: int = Query(5, ge=1, le=20)
):
    if interval not in ("hour", "day"):
        raise HTTPException(status_code=422, detail='interval must be "hour" or "day"')

    data = compute_nowcast_attention(
        LOGS_DIR / "retraining_log.jsonl",
        LOGS_DIR / "retraining_triggered.jsonl",
        days=days, interval=interval, lookback=lookback, z_cap=z_cap, top=top
    )
    if (not data.get("origins")) and os.getenv("DEMO_MODE", "false").lower() in ("1","true","yes"):
        data = _demo_seed(days, interval, top)
    return data
