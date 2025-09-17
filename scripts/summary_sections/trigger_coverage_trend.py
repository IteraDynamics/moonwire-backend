# scripts/summary_sections/trigger_coverage_trend.py
from __future__ import annotations
import os, json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Iterable
from collections import defaultdict, OrderedDict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from .common import SummaryContext, parse_ts


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _iter_log_jsonl(logs_dir: Path) -> Iterable[Path]:
    # Be liberal: any .jsonl under logs/ may contain candidates
    if not logs_dir.exists():
        return []
    return logs_dir.rglob("*.jsonl")


def _floor_bucket(ts: datetime, bucket_h: int) -> datetime:
    base = ts.replace(minute=0, second=0, microsecond=0)
    # snap to multiples of bucket_h
    delta_h = (base.hour % bucket_h)
    return base - timedelta(hours=delta_h)


def _band_class(rate: float, cand: int, trig: int) -> Tuple[str, str]:
    """Return (class, emoji) using requested rules."""
    if cand < 3 or trig < 3:
        return ("Insufficient", "ℹ️")
    if rate >= 0.15:
        return ("High", "✅")
    if rate >= 0.05:
        return ("Medium", "⚠️")
    return ("Low", "❌")


def _collect_counts(
    logs_dir: Path,
    triggers_path: Path,
    window_h: int,
    bucket_h: int,
) -> Tuple[Dict[str, Dict[datetime, Dict[str, int]]], List[datetime]]:
    """Return per-origin, per-bucket counts and the ordered bucket timeline."""
    now = datetime.now(timezone.utc)
    t_start = now - timedelta(hours=window_h)

    # Build bucket sequence
    buckets: List[datetime] = []
    t = _floor_bucket(t_start, bucket_h)
    end_bucket = _floor_bucket(now, bucket_h)
    while t <= end_bucket:
        buckets.append(t)
        t += timedelta(hours=bucket_h)

    # per-origin -> per-bucket -> counts
    counts: Dict[str, Dict[datetime, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"cand": 0, "trig": 0}))

    # Candidates from logs/
    for p in _iter_log_jsonl(logs_dir):
        for r in _load_jsonl(p):
            ts = parse_ts(r.get("timestamp"))
            if not ts or ts < t_start or ts > now:
                continue
            origin = r.get("origin") or r.get("source") or "unknown"
            if origin == "unknown":
                continue
            b = _floor_bucket(ts, bucket_h)
            counts[origin][b]["cand"] += 1

    # Triggers from models/
    for r in _load_jsonl(triggers_path):
        ts = parse_ts(r.get("timestamp"))
        if not ts or ts < t_start or ts > now:
            continue
        origin = r.get("origin") or "unknown"
        if origin == "unknown":
            continue
        b = _floor_bucket(ts, bucket_h)
        counts[origin][b]["trig"] += 1

    # Ensure every known origin has all buckets initialized
    for origin, per_b in list(counts.items()):
        for b in buckets:
            per_b.setdefault(b, {"cand": 0, "trig": 0})

    return counts, buckets


def _maybe_seed_demo_series(
    counts: Dict[str, Dict[datetime, Dict[str, int]]],
    buckets: List[datetime],
    is_demo: bool,
) -> Tuple[Dict[str, Dict[datetime, Dict[str, int]]], bool]:
    """If there is little/no data and DEMO_MODE is on, synthesize plausible series."""
    if not is_demo:
        return counts, False
    total = sum(v["cand"] + v["trig"] for o in counts.values() for v in o.values())
    if total >= 6:
        return counts, False  # we have enough real data

    if not buckets:
        now = datetime.now(timezone.utc)
        buckets = [_floor_bucket(now - timedelta(hours=6), 3),
                   _floor_bucket(now - timedelta(hours=3), 3),
                   _floor_bucket(now, 3)]

    synth = {
        "twitter": [ (20, 4), (25, 5), (22, 4) ],    # (cand, trig)
        "reddit":  [ (18, 2), (16, 1), (20, 2) ],
        "rss_news":[ (30, 1), (28, 2), (26, 1) ],
    }
    new_counts: Dict[str, Dict[datetime, Dict[str, int]]] = defaultdict(dict)
    for origin, pairs in synth.items():
        for b, (c, t) in zip(buckets[-len(pairs):], pairs):
            new_counts[origin][b] = {"cand": c, "trig": t}
        # backfill missing buckets with zeros
        for b in buckets:
            new_counts[origin].setdefault(b, {"cand": 0, "trig": 0})

    return new_counts, True


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Render a trigger coverage trend chart per origin and append a single markdown line
    linking to the artifact.
    """
    window_h = int(os.getenv("MW_TRIGGER_COVERAGE_WINDOW_H", "48"))
    bucket_h = int(os.getenv("MW_TRIGGER_BUCKET_H", "3"))

    img_name = f"trigger_coverage_trend_{window_h}h.png"
    img_path = Path("artifacts") / img_name
    img_path.parent.mkdir(parents=True, exist_ok=True)

    triggers_path = ctx.models_dir / "trigger_history.jsonl"
    counts, buckets = _collect_counts(ctx.logs_dir, triggers_path, window_h, bucket_h)

    # Demo synth if empty/sparse and DEMO_MODE
    counts, demo_seeded = _maybe_seed_demo_series(counts, buckets, ctx.is_demo)

    # Early out if totally empty
    if not counts:
        md.append(f"### 📈 Trigger Coverage Trend ({window_h}h)")
        md.append("_no data available_")
        return

    # Plot
    fig = plt.figure(figsize=(9, 3))
    ax = plt.gca()

    # Shaded quality bands
    ax.axhspan(0.15, 1.0, alpha=0.08)  # High zone
    ax.axhspan(0.05, 0.15, alpha=0.08) # Medium zone
    ax.axhspan(0.0, 0.05, alpha=0.08)  # Low zone

    bucket_labels = [b.strftime("%m-%d %H:%M") for b in buckets]

    # One line per origin
    for origin in sorted(counts.keys()):
        ys: List[float] = []
        for b in buckets:
            c = counts[origin].get(b, {"cand": 0, "trig": 0})
            rate = c["trig"] / max(c["cand"], 1)
            ys.append(rate)
        ax.plot(bucket_labels, ys, marker="o", linewidth=1.5, label=origin)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("trigger rate")
    title = f"Trigger Coverage Trend ({window_h}h)" + (" [demo]" if demo_seeded else "")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(img_path, dpi=150)
    plt.close(fig)

    # Markdown line
    md.append(f"📈 Trigger Coverage Trend ({window_h}h): {img_path}")