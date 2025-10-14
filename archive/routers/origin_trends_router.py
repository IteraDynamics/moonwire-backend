from __future__ import annotations
import os, random
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Query
from src.analytics.origin_trends import compute_origin_trends
from src.paths import LOGS_DIR

router = APIRouter(prefix="/internal", tags=["internal"])


def _demo_seed_trends(days: int, interval: str):
    """Seed 3 origins with plausible buckets when DEMO_MODE and no data."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    def buckets_for(days, interval):
        out = []
        if interval == "day":
            for i in range(days):
                ts = (now - timedelta(days=(days - 1 - i))).replace(hour=0)
                out.append({
                    "timestamp_bucket": ts.isoformat(),
                    "flags_count": random.randint(0, 8),
                    "triggers_count": random.randint(0, 4),
                })
        else:  # hour
            total = days * 24
            for i in range(total):
                ts = now - timedelta(hours=(total - 1 - i))
                out.append({
                    "timestamp_bucket": ts.isoformat(),
                    "flags_count": random.randint(0, 2),
                    "triggers_count": random.randint(0, 1),
                })
        return out

    origins = []
    for o in sorted(["reddit", "rss_news", "twitter"]):
        origins.append({"origin": o, "buckets": buckets_for(days, interval)})
    return {"window_days": days, "interval": interval, "origins": origins, "notes": ["demo seeded"]}


@router.get("/origin-trends")
def origin_trends(
    days: int = Query(7, ge=1),
    interval: str = Query("day")
):
    if interval not in ("day", "hour"):
        raise HTTPException(status_code=400, detail='interval must be "day" or "hour"')

    data = compute_origin_trends(
        LOGS_DIR / "retraining_log.jsonl",
        LOGS_DIR / "retraining_triggered.jsonl",
        days=days,
        interval=interval,
    )
    if (not data["origins"]) and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        data = _demo_seed_trends(days, interval)
    return data