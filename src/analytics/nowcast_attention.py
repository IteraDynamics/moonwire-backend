from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, DefaultDict

from src.analytics.origin_utils import stream_jsonl, extract_origin, parse_ts
from src.analytics.volatility_regimes import compute_volatility_regimes
from src.analytics.threshold_policy import threshold_for_regime
from src.analytics.source_metrics import compute_source_metrics
from src.analytics.lead_lag import compute_lead_lag


def _bucket_start(dt: datetime, interval: str) -> datetime:
    if interval == "day":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif interval == "hour":
        return dt.replace(minute=0, second=0, microsecond=0)
    else:
        raise ValueError('interval must be "hour" or "day"')


def _bucket_range(now: datetime, days: int, interval: str) -> List[datetime]:
    if interval == "day":
        start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        step = timedelta(days=1); count = days
    else:
        hours = days * 24
        start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours - 1)
        step = timedelta(hours=1); count = hours
    out, cur = [], start
    for _ in range(count):
        out.append(cur); cur = cur + step
    return out


def _mean_std(vals: List[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    return m, var ** 0.5


def compute_nowcast_attention(
    flags_path: Path,
    triggers_path: Path,
    days: int = 7,
    interval: str = "hour",
    lookback: int = 72,         # used for regimes; counts cover the full window
    z_cap: float = 5.0,         # cap for z normalization
    top: int = 10               # limit results
) -> Dict[str, Any]:
    """
    Blend components per origin into a single attention score for the most recent bucket:
      - burst z-score (current bucket vs window baseline, includes zeros)
      - volatility regime (mapped to factor via threshold policy)
      - precision prior (triggers/flags)
      - lead-lag leadership (max |r| where origin is currently the leader)

    Returns:
      {
        "window_days": days,
        "interval": "hour",
        "as_of": "ISO bucket",
        "origins": [
          {"origin": "twitter", "score": 87.4, "rank": 1,
           "components": {"z": 2.1, "z_norm": 0.42, "precision": 0.71,
                          "leadership": 0.58, "regime": "turbulent",
                          "regime_factor": 1.1, "threshold": 3.0}}
        ]
      }
    """
    if days <= 0:
        return {"window_days": days, "interval": interval, "as_of": None, "origins": []}
    if interval not in ("hour", "day"):
        raise ValueError('interval must be "hour" or "day"')
    if lookback <= 1:
        raise ValueError("lookback must be >= 2")
    if z_cap <= 0:
        z_cap = 1.0

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    buckets = _bucket_range(now, days, interval)
    last_bucket = buckets[-1]

    # Build combined counts per origin per bucket (flags + triggers)
    per_origin: DefaultDict[str, Dict[datetime, int]] = defaultdict(lambda: defaultdict(int))

    # Flags
    if flags_path.exists():
        for row in stream_jsonl(flags_path):
            ts = parse_ts(row.get("timestamp")) or now
            if ts < cutoff:
                continue
            bts = _bucket_start(ts, interval)
            origin = extract_origin(
                row.get("origin")
                or row.get("source")
                or (row.get("meta") or {}).get("origin")
                or (row.get("metadata") or {}).get("source")
            )
            per_origin[origin][bts] += 1

    # Triggers
    if triggers_path.exists():
        for row in stream_jsonl(triggers_path):
            ts = parse_ts(row.get("timestamp")) or now
            if ts < cutoff:
                continue
            bts = _bucket_start(ts, interval)
            origin = extract_origin(
                row.get("origin")
                or row.get("source")
                or (row.get("meta") or {}).get("origin")
                or (row.get("metadata") or {}).get("source")
            )
            per_origin[origin][bts] += 1

    origins = sorted(per_origin.keys())
    if not origins:
        return {"window_days": days, "interval": interval, "as_of": last_bucket.isoformat().replace("+00:00","Z"), "origins": []}

    # Components from other analytics
    regimes = compute_volatility_regimes(flags_path, triggers_path, days=days, interval=interval, lookback=lookback)
    regime_map = {r["origin"]: r.get("regime", "normal") for r in regimes.get("origins", [])}

    metrics = compute_source_metrics(flags_path, triggers_path, days=days, min_count=1)
    precision_map = {o["origin"]: float(o.get("precision", 0) if "precision" in o else (o.get("triggers", 0) / max(o.get("flags", 0), 1))) for o in metrics.get("origins", [])}

    ll = compute_lead_lag(flags_path, triggers_path, days=days, interval=interval, max_lag=24, use="flags")
    leadership_map: Dict[str, float] = defaultdict(float)
    for p in ll.get("pairs", []):
        # best_lag > 0 => a leads b, <0 => b leads a
        r = abs(float(p.get("correlation", 0)))
        if p.get("best_lag", 0) > 0:
            leadership_map[p["a"]] = max(leadership_map[p["a"]], r)
        elif p.get("best_lag", 0) < 0:
            leadership_map[p["b"]] = max(leadership_map[p["b"]], r)

    # Compute z-score for the last bucket and blend components
    out_rows: List[Dict[str, Any]] = []

    for origin in origins:
        series = [float(per_origin[origin].get(b, 0)) for b in buckets]  # include zeros for baseline
        mean, std = _mean_std(series)
        current = series[-1]
        z = (current - mean) / std if std > 0 else 0.0
        z_norm = max(0.0, min(z / z_cap, 1.0))  # clamp to [0,1]

        regime = regime_map.get(origin, "normal")
        threshold = threshold_for_regime(regime)
        # Regime factor: slight emphasis on turbulent, small de-emphasis on calm
        regime_factor = {"calm": 0.95, "normal": 1.0, "turbulent": 1.05}.get(regime, 1.0)

        precision = max(0.0, min(float(precision_map.get(origin, 0.0)), 1.0))
        leadership = max(0.0, min(float(leadership_map.get(origin, 0.0)), 1.0))

        # Blend (weights sum to 1.0)
        w_z, w_prec, w_lead, w_reg = 0.5, 0.25, 0.15, 0.10
        base = (w_z * z_norm) + (w_prec * precision) + (w_lead * leadership) + (w_reg * (regime_factor - 0.9) / 0.2)
        score = round(100.0 * base, 2)

        out_rows.append({
            "origin": origin,
            "score": score,
            "components": {
                "z": round(float(z), 3), "z_norm": round(float(z_norm), 3),
                "precision": round(precision, 3), "leadership": round(leadership, 3),
                "regime": regime, "regime_factor": regime_factor, "threshold": threshold,
                "mean": round(mean, 3), "std": round(std, 3), "current": int(current),
            }
        })

    # Rank and trim
    out_rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(out_rows, 1):
        r["rank"] = i
    if top and top > 0:
        out_rows = out_rows[:top]

    return {
        "window_days": days,
        "interval": interval,
        "as_of": last_bucket.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "origins": out_rows
    }
