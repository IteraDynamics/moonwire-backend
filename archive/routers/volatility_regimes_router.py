from __future__ import annotations
import os, random
from fastapi import APIRouter, HTTPException, Query
from src.analytics.volatility_regimes import compute_volatility_regimes
from src.analytics.threshold_policy import threshold_for_regime
from src.paths import LOGS_DIR

router = APIRouter(prefix="/internal", tags=["internal"])

def _demo_seed(days: int, interval: str):
    origins = ["twitter", "reddit", "rss_news"]
    regimes = ["turbulent", "normal", "calm"]
    random.shuffle(regimes)
    out = []
    for o, r in zip(origins, regimes):
        out.append({
            "origin": o,
            "vol_metric": round(random.uniform(0.1, 5.0), 3),
            "regime": r,
            "stats": {"mean": round(random.uniform(0.0, 3.0), 3), "std": round(random.uniform(0.0, 2.5), 3)}
        })
    return {"window_days": days, "interval": interval, "origins": out, "notes": ["demo regimes seeded"]}

@router.get("/volatility-regimes")
def volatility_regimes(
    days: int = Query(30, ge=1),
    interval: str = Query("hour"),
    lookback: int = Query(72, ge=2),
    q_calm: float = Query(0.33, ge=0.0, le=1.0),
    q_turb: float = Query(0.80, ge=0.0, le=1.0)
):
    if interval not in ("hour", "day"):
        raise HTTPException(status_code=422, detail='interval must be "hour" or "day"')

    data = compute_volatility_regimes(
        flags_path=LOGS_DIR / "retraining_log.jsonl",
        triggers_path=LOGS_DIR / "retraining_triggered.jsonl",
        days=days,
        interval=interval,
        lookback=lookback,
        q_calm=q_calm,
        q_turb=q_turb,
    )

    if (not data.get("origins")) and os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes"):
        data = _demo_seed(days, interval)

    return data

@router.get("/adaptive-thresholds")
def adaptive_thresholds(
    days: int = Query(30, ge=1),
    interval: str = Query("hour"),
    lookback: int = Query(72, ge=2),
    q_calm: float = Query(0.33, ge=0.0, le=1.0),
    q_turb: float = Query(0.80, ge=0.0, le=1.0)
):
    vr = volatility_regimes(days, interval, lookback, q_calm, q_turb)  # call handler above
    origins = []
    for row in vr["origins"]:
        regime = row.get("regime", "normal")
        origins.append({
            "origin": row.get("origin"),
            "regime": regime,
            "threshold": threshold_for_regime(regime)
        })
    return {"interval": vr["interval"], "origins": origins}
