from __future__ import annotations
import os, random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, DefaultDict

from src.analytics.origin_utils import stream_jsonl, extract_origin, parse_ts

# Optional helpers from your analytics (used once per run to avoid heavy recompute per-bucket)
from src.analytics.source_metrics import compute_source_metrics
from src.analytics.lead_lag import compute_lead_lag

@dataclass
class FeatureRow:
    ts: datetime
    origin: str
    x: List[float]
    y: int

FEATURE_ORDER = [
    "count_1h", "count_6h", "count_24h", "count_72h",
    "burst_z",
    "regime_calm", "regime_normal", "regime_turbulent",
    "precision_7d", "recall_7d",
    "leadership_max_r",
]

def _bucket_start(dt: datetime, interval: str) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0) if interval == "hour" else dt.replace(hour=0, minute=0, second=0, microsecond=0)

def _bucket_range(now: datetime, days: int, interval: str) -> List[datetime]:
    if interval == "day":
        start = (now - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
        step, n = timedelta(days=1), days
    else:
        hours = days * 24
        start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours - 1)
        step, n = timedelta(hours=1), hours
    out, cur = [], start
    for _ in range(n):
        out.append(cur); cur += step
    return out

def _mean_std(vals: List[float]) -> Tuple[float, float]:
    n = len(vals); 
    if n == 0: return 0.0, 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    return m, var ** 0.5

def _rolling_sum(arr: List[int], win: int) -> List[int]:
    if win <= 1: return arr[:]
    out, s = [], 0
    for i, v in enumerate(arr):
        s += v
        if i >= win: s -= arr[i - win]
        out.append(s if i >= win - 1 else sum(arr[:i+1]))
    return out

def _class_balance(rows: List[FeatureRow], ratio: float = 4.0) -> List[FeatureRow]:
    pos = [r for r in rows if r.y == 1]
    neg = [r for r in rows if r.y == 0]
    if not pos or not neg: return rows
    # downsample majority if > ratio:1
    maj, minr = (neg, pos) if len(neg) > len(pos) else (pos, neg)
    if len(maj) <= ratio * len(minr): return rows
    random.seed(42)
    k = int(ratio * len(minr))
    maj_ds = random.sample(maj, k)
    return (maj_ds + minr) if maj is neg else (neg + maj_ds)

def _load_counts(flags_path: Path, triggers_path: Path, days: int, interval: str) -> Tuple[Dict[str, Dict[datetime, int]], Dict[str, Dict[datetime, int]], List[datetime]]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    buckets = _bucket_range(now, days, interval)
    per_origin_flags: DefaultDict[str, Dict[datetime, int]] = defaultdict(lambda: defaultdict(int))
    per_origin_trig:  DefaultDict[str, Dict[datetime, int]] = defaultdict(lambda: defaultdict(int))

    if flags_path.exists():
        for row in stream_jsonl(flags_path):
            ts = parse_ts(row.get("timestamp")) or now
            if ts < cutoff: continue
            b = _bucket_start(ts, interval)
            o = extract_origin(row.get("origin") or row.get("source") or (row.get("meta") or {}).get("origin") or (row.get("metadata") or {}).get("source"))
            per_origin_flags[o][b] += 1
    if triggers_path.exists():
        for row in stream_jsonl(triggers_path):
            ts = parse_ts(row.get("timestamp")) or now
            if ts < cutoff: continue
            b = _bucket_start(ts, interval)
            o = extract_origin(row.get("origin") or row.get("source") or (row.get("meta") or {}).get("origin") or (row.get("metadata") or {}).get("source"))
            per_origin_trig[o][b] += 1

    return per_origin_flags, per_origin_trig, buckets

def _once_precision_recall(flags_path: Path, triggers_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    m = compute_source_metrics(flags_path, triggers_path, days=7, min_count=1)
    prec = {o["origin"]: float(o.get("precision", 0.0)) for o in m.get("origins", [])}
    rec  = {o["origin"]: float(o.get("recall", 0.0))    for o in m.get("origins", [])}
    return prec, rec

def _once_leadership(flags_path: Path, triggers_path: Path, interval: str) -> Dict[str, float]:
    ll = compute_lead_lag(flags_path, triggers_path, days=7, interval=interval, max_lag=24, use="flags")
    lead: Dict[str, float] = defaultdict(float)
    for p in ll.get("pairs", []):
        r = abs(float(p.get("correlation", 0)))
        if p.get("best_lag", 0) > 0:
            lead[p["a"]] = max(lead[p["a"]], r)
        elif p.get("best_lag", 0) < 0:
            lead[p["b"]] = max(lead[p["b"]], r)
    return lead

def build_examples(flags_path: Path, triggers_path: Path, days: int = 14, interval: str = "hour") -> Tuple[List[FeatureRow], List[str]]:
    """Build hourly examples; y=1 if any triggers for that origin in (t, t+6h]."""
    if interval not in ("hour", "day"): raise ValueError('interval must be "hour" or "day"')
    per_flags, per_trig, buckets = _load_counts(flags_path, triggers_path, days, interval)
    if not per_flags and not per_trig:
        return [], FEATURE_ORDER[:]  # handled by demo later

    # “Global” helpers (computed once per run)
    precision_7d, recall_7d = _once_precision_recall(flags_path, triggers_path)
    leadership = _once_leadership(flags_path, triggers_path, interval)

    rows: List[FeatureRow] = []
    # simple cross-origin regime buckets using rolling std over last 72 buckets (approx)
    look = 72 if interval == "hour" else 3  # small for daily
    for i, ts in enumerate(buckets):
        if i == 0 or i + 6 >= len(buckets):  # need future 6 for label
            continue
        # rolling std up to (and incl) i-1 (no leakage)
        std_by_origin: Dict[str, float] = {}
        for o in set(list(per_flags.keys()) + list(per_trig.keys())):
            series = [float(per_flags[o].get(b, 0) + per_trig[o].get(b, 0)) for b in buckets[:i]]
            window = series[-look:] if len(series) >= look else series
            _, sd = _mean_std(window)
            std_by_origin[o] = sd
        # regime thresholds (cross-origin quantiles)
        xs = sorted(std_by_origin.values()) if std_by_origin else [0.0]
        q_lo = xs[int(0.33 * (len(xs) - 1))] if xs else 0.0
        q_hi = xs[int(0.80 * (len(xs) - 1))] if xs else 0.0

        for o in sorted(std_by_origin.keys()):
            # features at time i
            seq = [int(per_flags[o].get(b, 0) + per_trig[o].get(b, 0)) for b in buckets[:i+1]]
            c1  = seq[-1]
            c6  = sum(seq[max(0, len(seq)-6):])
            c24 = sum(seq[max(0, len(seq)-24):])  if interval == "hour" else sum(seq)
            c72 = sum(seq[max(0, len(seq)-72):])  if interval == "hour" else sum(seq)

            mean, sd = _mean_std(seq[:-1] or [0.0])
            z = (c1 - mean) / sd if sd > 0 else 0.0

            std_now = std_by_origin[o]
            regime = "calm" if std_now <= q_lo else ("turbulent" if std_now >= q_hi else "normal")
            one_hot = (1.0, 0.0, 0.0) if regime == "calm" else ((0.0, 1.0, 0.0) if regime == "normal" else (0.0, 0.0, 1.0))

            prec = precision_7d.get(o, 0.0)
            rec  = recall_7d.get(o, 0.0)
            lead = leadership.get(o, 0.0)

            # label: any triggers in next 6 buckets
            future_bs = buckets[i+1:i+7]
            y = 1 if any(per_trig[o].get(b, 0) > 0 for b in future_bs) else 0

            x = [float(c1), float(c6), float(c24), float(c72), float(z), *one_hot, float(prec), float(rec), float(lead)]
            rows.append(FeatureRow(ts=buckets[i], origin=o, x=x, y=y))

    # Balance if skewed
    rows = _class_balance(rows, ratio=4.0)
    return rows, FEATURE_ORDER[:]

def build_feature_row_for(flags_path: Path, triggers_path: Path, origin: str, ts: datetime, interval: str = "hour") -> Tuple[Dict[str, float], List[str]]:
    """Build one row’s features for {origin, ts} using only history <= ts."""
    days = 14
    per_flags, per_trig, buckets = _load_counts(flags_path, triggers_path, days, interval)
    if not buckets: return {}, FEATURE_ORDER[:]
    ts = _bucket_start(ts.astimezone(timezone.utc), interval)
    if ts not in buckets:  # snap to nearest past bucket
        buckets.append(ts); buckets.sort()
    i = buckets.index(ts)
    precision_7d, recall_7d = _once_precision_recall(flags_path, triggers_path)
    leadership = _once_leadership(flags_path, triggers_path, interval)

    # rolling std cross-origin (as above)
    look = 72 if interval == "hour" else 3
    std_by_origin = {}
    for o in set(list(per_flags.keys()) + list(per_trig.keys())):
        series = [float(per_flags[o].get(b, 0) + per_trig[o].get(b, 0)) for b in buckets[:i]]
        window = series[-look:] if len(series) >= look else series
        _, sd = _mean_std(window)
        std_by_origin[o] = sd
    xs = sorted(std_by_origin.values()) or [0.0]
    q_lo = xs[int(0.33 * (len(xs) - 1))]
    q_hi = xs[int(0.80 * (len(xs) - 1))]
    # this origin’s features
    o = origin
    seq = [int(per_flags[o].get(b, 0) + per_trig[o].get(b, 0)) for b in buckets[:i+1]]
    c1  = seq[-1]
    c6  = sum(seq[max(0, len(seq)-6):])
    c24 = sum(seq[max(0, len(seq)-24):]) if interval == "hour" else sum(seq)
    c72 = sum(seq[max(0, len(seq)-72):]) if interval == "hour" else sum(seq)
    mean, sd = _mean_std(seq[:-1] or [0.0])
    z = (c1 - mean) / sd if sd > 0 else 0.0
    std_now = std_by_origin.get(o, 0.0)
    regime = "calm" if std_now <= q_lo else ("turbulent" if std_now >= q_hi else "normal")
    one_hot = (1.0, 0.0, 0.0) if regime == "calm" else ((0.0, 1.0, 0.0) if regime == "normal" else (0.0, 0.0, 1.0))
    prec = precision_7d.get(o, 0.0); rec = recall_7d.get(o, 0.0); lead = leadership.get(o, 0.0)

    feats = {
        "count_1h": float(c1), "count_6h": float(c6), "count_24h": float(c24), "count_72h": float(c72),
        "burst_z": float(z), "regime_calm": one_hot[0], "regime_normal": one_hot[1], "regime_turbulent": one_hot[2],
        "precision_7d": float(prec), "recall_7d": float(rec), "leadership_max_r": float(lead),
    }
    return feats, FEATURE_ORDER[:]

# --- Demo seeding if no rows at all ---
def synth_demo_dataset() -> Tuple[List[FeatureRow], List[str], Dict[str, Any]]:
    random.seed(42)
    rows: List[FeatureRow] = []
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    origins = ["twitter", "reddit", "rss_news"]
    for o in origins:
        for h in range(120):  # 5 days
            ts = now - timedelta(hours=120 - h)
            # create burst_z correlated with label
            z = random.random() * 3.0
            y = 1 if z + random.random() * 0.5 > 2.2 else 0
            x = [0, 0, 0, 0, z, 0, 1, 0, 0.6, 0.3, 0.3]  # simple baseline features
            rows.append(FeatureRow(ts=ts, origin=o, x=x, y=y))
    return rows, FEATURE_ORDER[:], {"demo": True}
