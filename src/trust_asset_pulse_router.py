from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import json

router = APIRouter(prefix="/internal", tags=["internal-tools"])

SIGNAL_HISTORY_PATH = Path("data/signal_history.jsonl")
SUPPRESSION_LOG_PATH = Path("data/suppression_log.jsonl")
RETRAIN_QUEUE_PATH = Path("data/retrain_queue.jsonl")
OVERRIDE_LOG_PATH = Path("data/override_log.jsonl")


def parse_jsonl(path, start_dt=None, end_dt=None, asset_filter=None):
    entries = []
    if not path.exists():
        return entries

    with path.open("r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                ts = entry.get("timestamp")
                asset = entry.get("asset")
                if not ts or not asset:
                    continue
                ts_dt = datetime.fromisoformat(ts)
                if start_dt and ts_dt < start_dt:
                    continue
                if end_dt and ts_dt > end_dt:
                    continue
                if asset_filter and asset != asset_filter:
                    continue
                entries.append(entry)
            except Exception:
                continue
    return entries


@router.get("/trust-asset-pulse")
def trust_asset_pulse(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    asset: Optional[str] = Query(None),
    order_by: Optional[str] = Query(None)
):
    now = datetime.utcnow()
    start_dt = datetime.fromisoformat(start_date) if start_date else now - timedelta(hours=24)
    end_dt = datetime.fromisoformat(end_date) if end_date else now

    history = parse_jsonl(SIGNAL_HISTORY_PATH, start_dt, end_dt, asset)
    suppressions = parse_jsonl(SUPPRESSION_LOG_PATH, start_dt, end_dt, asset)
    retrains = parse_jsonl(RETRAIN_QUEUE_PATH, start_dt, end_dt, asset)
    overrides = parse_jsonl(OVERRIDE_LOG_PATH, start_dt, end_dt, asset)

    trust_by_asset = defaultdict(list)
    for entry in history:
        trust_by_asset[entry["asset"]].append(entry.get("trust_score", 0.5))

    suppression_count = Counter([s["asset"] for s in suppressions])
    retrain_count = Counter([r["asset"] for r in retrains])
    override_count = Counter([o["asset"] for o in overrides])
    retrain_hint_map = defaultdict(list)
    trust_trends = {}

    for r in retrains:
        retrain_hint_map[r["asset"]].append(r.get("retrain_hint", "none"))

    for asset_key, scores in trust_by_asset.items():
        if len(scores) >= 2:
            trend = "flat"
            if scores[-1] > scores[0]:
                trend = "up"
            elif scores[-1] < scores[0]:
                trend = "down"
            trust_trends[asset_key] = trend

    results = []
    for asset_key in trust_by_asset:
        scores = trust_by_asset[asset_key]
        avg_trust = round(sum(scores) / len(scores), 3) if scores else 0.5
        retrain_hints = retrain_hint_map.get(asset_key, [])
        most_common_hint = Counter(retrain_hints).most_common(1)
        results.append({
            "asset": asset_key,
            "avg_trust_score": avg_trust,
            "suppressed_count": suppression_count.get(asset_key, 0),
            "retrained_count": retrain_count.get(asset_key, 0),
            "override_count": override_count.get(asset_key, 0),
            "most_common_retrain_hint": most_common_hint[0][0] if most_common_hint else None,
            "trust_trend": trust_trends.get(asset_key, "flat")
        })

    if order_by and results:
        if order_by in results[0]:
            results = sorted(results, key=lambda x: x[order_by], reverse=True)

    return {
        "window": {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat()
        },
        "results": results
    }
