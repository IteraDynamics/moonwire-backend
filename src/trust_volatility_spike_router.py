from fastapi import APIRouter, Query
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os

router = APIRouter(prefix="/internal")

SIGNAL_HISTORY_PATH = "data/signal_history.jsonl"
SUPPRESSION_LOG_PATH = "data/suppression_log.jsonl"
RETRAIN_QUEUE_PATH = "data/retrain_queue.jsonl"

def load_jsonl_between(path, start_dt, end_dt):
    entries = []
    if not os.path.exists(path):
        return entries

    with open(path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                timestamp = datetime.fromisoformat(entry.get("timestamp"))
                if start_dt <= timestamp <= end_dt:
                    entries.append(entry)
            except Exception:
                continue
    return entries

def compute_avg_trust(entries):
    asset_scores = defaultdict(list)
    for e in entries:
        asset = e.get("asset")
        trust = e.get("trust_score")
        if asset and isinstance(trust, (int, float)):
            asset_scores[asset].append(trust)

    return {a: sum(scores)/len(scores) for a, scores in asset_scores.items() if scores}

@router.get("/trust-volatility-spikes")
def get_trust_volatility_spikes(
    start_date: str = Query(...),
    end_date: str = Query(...),
    min_spike_delta: float = Query(0.3)
):
    try:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
    except Exception:
        return {"error": "Invalid date format. Use ISO8601."}

    mid_dt = start_dt + (end_dt - start_dt) / 2

    # Split window: early and late
    early_signals = load_jsonl_between(SIGNAL_HISTORY_PATH, start_dt, mid_dt)
    late_signals = load_jsonl_between(SIGNAL_HISTORY_PATH, mid_dt, end_dt)

    early_suppressions = load_jsonl_between(SUPPRESSION_LOG_PATH, start_dt, mid_dt)
    late_suppressions = load_jsonl_between(SUPPRESSION_LOG_PATH, mid_dt, end_dt)

    early_retrains = load_jsonl_between(RETRAIN_QUEUE_PATH, start_dt, mid_dt)
    late_retrains = load_jsonl_between(RETRAIN_QUEUE_PATH, mid_dt, end_dt)

    early_avg = compute_avg_trust(early_signals)
    late_avg = compute_avg_trust(late_signals)

    all_assets = set(early_avg.keys()) | set(late_avg.keys())

    def count_by_asset(entries):
        count_map = defaultdict(int)
        for e in entries:
            if e.get("asset"):
                count_map[e["asset"]] += 1
        return count_map

    early_supp = count_by_asset(early_suppressions)
    late_supp = count_by_asset(late_suppressions)
    early_retrain = count_by_asset(early_retrains)
    late_retrain = count_by_asset(late_retrains)

    results = []
    for asset in all_assets:
        start_score = early_avg.get(asset, 0)
        end_score = late_avg.get(asset, 0)
        delta = end_score - start_score

        if abs(delta) >= min_spike_delta:
            results.append({
                "asset": asset,
                "avg_trust_score_start": round(start_score, 4),
                "avg_trust_score_end": round(end_score, 4),
                "delta": round(delta, 4),
                "spike_direction": "up" if delta > 0 else "down",
                "suppression_count_change": late_supp.get(asset, 0) - early_supp.get(asset, 0),
                "retrain_flag_change": late_retrain.get(asset, 0) - early_retrain.get(asset, 0)
            })

    results.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return {
        "window": {
            "start": start_date,
            "end": end_date
        },
        "results": results[:10]
    }