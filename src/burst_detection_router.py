from __future__ import annotations
import os, random
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Query
from src.analytics.burst_detection import compute_bursts
from src.paths import LOGS_DIR

# Use an internal prefix here for robustness (matches the rest of your internal APIs)
router = APIRouter(prefix="/internal", tags=["internal"])

def _demo_seed_bursts(days: int, interval: str, z_thresh: float):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    step = timedelta(hours=1) if interval == "hour" else timedelta(days=1)
    origins = ["twitter", "reddit", "rss_news"]
    out = []
    for o in origins[:random.randint(2, 3)]:
        k = random.randint(1, 3)
        bursts = []
        t0 = now - (days * (24 if interval == "hour" else 1)) * step + 10 * step
        for i in range(k):
            ts = t0 + random.randint(0, 10) * step
            bursts.append({
                "timestamp_bucket": ts.isoformat().replace("+00:00", "Z"),
                "count": random.randint(20, 60),
                "z_score": round(random.uniform(max(z_thresh, 2.0), max(z_thresh + 1.0, 4.0)), 2)
            })
        bursts.sort(key=lambda b: b["timestamp_bucket"])
        out.append({"origin": o, "bursts": bursts})
    return {"window_days": days, "interval": interval, "origins": out, "notes": ["demo bursts seeded"]}

@router.get("/burst-detection")
def burst_detection(
    days: int = Query(7, ge=1),
    interval: str = Query("hour"),
    z_thresh: float = Query(2.0)
):
    if interval not in ("hour", "day"):
        raise HTTPException(status_code=422, detail='interval must be "hour" or "day"')

    data = compute_bursts(
        flags_path=LOGS_DIR / "retraining_log.jsonl",
        triggers_path=LOGS_DIR / "retraining_triggered.jsonl",
        days=days,
        interval=interval,
        z_thresh=z_thresh
    )

    # Demo fallback: seed if completely empty (no bursts across all origins)
    has_any = any(len(o.get("bursts", [])) > 0 for o in data.get("origins", []))
    if (not has_any) and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        data = _demo_seed_bursts(days, interval, z_thresh)

    return data
