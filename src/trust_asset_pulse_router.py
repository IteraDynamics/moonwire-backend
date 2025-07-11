from fastapi import APIRouter, Query
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Optional
from pathlib import Path
import json

router = APIRouter()

# === File Paths ===
HISTORY_PATH = Path("data/signal_history.jsonl")
SUPPRESSION_PATH = Path("data/suppression_log.jsonl")
RETRAIN_PATH = Path("data/retrain_queue.jsonl")
OVERRIDE_PATH = Path("data/override_log.jsonl")


def load_jsonl(path: Path):
    if not path.exists():
        return []
    records = []
    with path.open("r") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def parse_ts(entry):
    try:
        return datetime.fromisoformat(entry["timestamp"])
    except Exception:
        return None


@router.get("/internal/trust-asset-pulse")
def trust_asset_pulse(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    asset: Optional[str] = Query(None),
    order_by: Optional[str] = Query("suppressed_count")
):
    # === Time Range ===
    now = datetime.utcnow()
    start_dt = datetime.fromisoformat(start_date) if start_date else now - timedelta(hours=24)
    end_dt = datetime.fromisoformat(end_date) if end_date else now

    # === Load + Filter ===
    history = [e for e in load_jsonl(HISTORY_PATH) if (ts := parse_ts(e)) and start_dt <= ts <= end_dt]
    suppressions = [e for e in load_jsonl(SUPPRESSION_PATH) if (ts := parse_ts(e)) and start_dt <= ts <= end_dt]
    retrains = [e for e in load_jsonl(RETRAIN_PATH) if (ts := parse_ts(e)) and start_dt <= ts <= end_dt]
    overrides = [e for e in load_jsonl(OVERRIDE_PATH) if (ts := parse_ts(e)) and start_dt <= ts <= end_dt]

    if asset:
        history = [e for e in history if e.get("asset") == asset]
        suppressions = [e for e in suppressions if e.get("asset") == asset]
        retrains = [e for e in retrains if e.get("asset") == asset]
        overrides = [e for e in overrides if e.get("asset") == asset]

    # === Aggregate ===
    asset_stats = defaultdict(lambda: {
        "trust_scores": [],
        "suppressed_count": 0,
        "retrained_count": 0,
        "override_count": 0,
        "retrain_hints": []
    })

    for e in history:
        a = e.get("asset", "unknown")
        asset_stats[a]["trust_scores"].append(e.get("trust_score", 0.5))

    for e in suppressions:
        a = e.get("asset", "unknown")
        asset_stats[a]["suppressed_count"] += 1

    for e in retrains:
        a = e.get("asset", "unknown")
        asset_stats[a]["retrained_count"] += 1
        hint = e.get("retrain_hint")
        if hint:
            asset_stats[a]["retrain_hints"].append(hint)

    for e in overrides:
        a = e.get("asset", "unknown")
        asset_stats[a]["override_count"] += 1

    # === Build Results ===
    results = []
    for asset_key, stats in asset_stats.items():
        avg_score = round(sum(stats["trust_scores"]) / len(stats["trust_scores"]), 3) if stats["trust_scores"] else 0.5
        hint_counts = Counter(stats["retrain_hints"])
        common_hint = hint_counts.most_common(1)[0][0] if hint_counts else None

        results.append({
            "asset": asset_key,
            "avg_trust_score": avg_score,
            "suppressed_count": stats["suppressed_count"],
            "retrained_count": stats["retrained_count"],
            "override_count": stats["override_count"],
            "most_common_retrain_hint": common_hint
        })

    if order_by and order_by in results[0]:
        results.sort(key=lambda x: x[order_by], reverse=True)

    return {"window": {"start": start_dt.isoformat(), "end": end_dt.isoformat()}, "results": results}
