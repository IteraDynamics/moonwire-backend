# scripts/summary_sections/trigger_coverage_trend.py
from __future__ import annotations

import os
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Iterable
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless for CI
import matplotlib.pyplot as plt

from .common import SummaryContext, parse_ts


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                # best effort
                continue
    return out


def _iter_log_jsonl(logs_dir: Path) -> Iterable[Path]:
    if not logs_dir.exists():
        return []
    return logs_dir.rglob("*.jsonl")


def _floor_bucket(ts: datetime, bucket_h: int) -> datetime:
    base = ts.replace(minute=0, second=0, microsecond=0)
    delta_h = base.hour % bucket_h
    return base - timedelta(hours=delta_h)


def _collect_counts(
    logs_dir: Path,
    triggers_path: Path,
    window_h: int,
    bucket_h: int,
) -> Tuple[Dict[str, Dict[datetime, Dict[str, int]]], List[datetime]]:
    """Return (counts, bucket_list).
    counts[origin][bucket] = {"cand": int, "trig": int}
    """
    now = datetime.now(timezone.utc)
    t_start = now - timedelta(hours=window_h)

    # build bucket sequence
    buckets: List[datetime] = []
    t = _floor_bucket(t_start, bucket_h)
    end = _floor_bucket(now, bucket_h)
    while t <= end:
        buckets.append(t)
        t += timedelta(hours=bucket_h)

    counts: Dict[str, Dict[datetime, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"cand": 0, "trig": 0}))

    # candidates from logs/
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

    # triggers from models/
    for r in _load_jsonl(triggers_path):
        ts = parse_ts(r.get("timestamp"))
        if not ts or ts < t_start or ts > now:
            continue
        origin = r.get("origin") or "unknown"
        if origin == "unknown":
            continue
        b = _floor_bucket(ts, bucket_h)
        counts[origin][b]["trig"] += 1

    # ensure each origin has all buckets
    for origin, per_b in list(counts.items()):
        for b in buckets:
            per_b.setdefault(b, {"cand": 0, "trig": 0})

    return counts, buckets


def _maybe_seed_demo_series(
    counts: Dict[str, Dict[datetime, Dict[str, int]]],
    buckets: List[datetime],
    is_demo: bool,
) -> Tuple[Dict[str, Dict[datetime, Dict[str, int]]], bool]:
    if not is_demo:
        return counts, False
    total = sum(c["cand"] + c["trig"] for per_b in counts.values() for c in per_b.values())
    if total >= 6:
        return counts, False

    # if no buckets provided, synthesize 3 recent ones
    if not buckets:
        now = datetime.now(timezone.utc)
        buckets = [
            _floor_bucket(now - timedelta(hours=6), 3),
            _floor_bucket(now - timedelta(hours=3), 3),
            _floor_bucket(now, 3),
        ]

    synth = {
        "twitter":  [(20, 4), (25, 5), (22, 4)],  # (cand, trig)
        "reddit":   [(18, 2), (16, 1), (20, 2)],
        "rss_news": [(30, 1), (28, 2), (26, 1)],
    }
    out: Dict[str, Dict[datetime, Dict[str, int]]] = defaultdict(dict)
    for origin, pairs in synth.items():
        for b, (cand, trig) in zip(buckets[-len(pairs):], pairs):
            out[origin][b] = {"cand": cand, "trig": trig}
        for b in buckets:
            out[origin].setdefault(b, {"cand": 0, "trig": 0})

    return out, True


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build and save the trigger coverage trend chart, then append a single line to the markdown.
    """
    window_h = int(os.getenv("MW_TRIGGER_COVERAGE_WINDOW_H", "48"))
    bucket_h = int(os.getenv("MW_TRIGGER_BUCKET_H", "3"))

    # Figure out all candidate artifact destinations:
    # - repo root: artifacts/
    # - alongside tests: <tmp>/artifacts (based on ctx.logs_dir / ctx.models_dir)
    img_name = f"trigger_coverage_trend_{window_h}h.png"
    candidate_dirs = [
        Path("artifacts"),
        getattr(ctx, "logs_dir", Path(".")) and Path(ctx.logs_dir).parent / "artifacts",
        getattr(ctx, "models_dir", Path(".")) and Path(ctx.models_dir).parent / "artifacts",
    ]
    # de-dup while preserving order
    seen = set()
    artifact_dirs = []
    for d in candidate_dirs:
        d = Path(d)
        key = d.resolve()
        if key not in seen:
            seen.add(key)
            artifact_dirs.append(d)

    # compute counts
    triggers_path = ctx.models_dir / "trigger_history.jsonl"
    counts, buckets = _collect_counts(ctx.logs_dir, triggers_path, window_h, bucket_h)
    counts, demo_seeded = _maybe_seed_demo_series(counts, buckets, ctx.is_demo)

    # Append header + early exit on no data
    md.append(f"📈 Trigger Coverage Trend ({window_h}h){' (demo)' if demo_seeded else ''}")

    if not counts or not buckets:
        md.append("_no data available_")
        return

    # Build plot
    fig = plt.figure(figsize=(9, 3))
    ax = plt.gca()

    # background bands
    ax.axhspan(0.15, 1.0, alpha=0.08)  # High
    ax.axhspan(0.05, 0.15, alpha=0.08) # Medium
    ax.axhspan(0.0, 0.05, alpha=0.08)  # Low

    xlabels = [b.strftime("%m-%d %H:%M") for b in buckets]

    for origin in sorted(counts.keys()):
        ys: List[float] = []
        for b in buckets:
            c = counts[origin].get(b, {"cand": 0, "trig": 0})
            rate = c["trig"] / max(c["cand"], 1)
            ys.append(rate)
        ax.plot(xlabels, ys, marker="o", linewidth=1.5, label=origin)

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("trigger rate")
    ax.set_title(f"Trigger Coverage Trend ({window_h}h)" + (" [demo]" if demo_seeded else ""))
    ax.legend(loc="upper left", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save to all artifact dirs to satisfy CI and tests
    last_path = None
    for d in artifact_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            p = d / img_name
            fig.savefig(p, dpi=150)
            last_path = p
        except Exception:
            # keep going; we only need one success
            continue
    plt.close(fig)

    # If nothing saved (unlikely), at least say so
    if last_path is None:
        md.append("_failed to write chart artifact_")
    else:
        # Standardize the link we print in CI to repo-root artifacts/
        md.append(f"- saved: {Path('artifacts') / img_name}")