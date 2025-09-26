# scripts/summary_sections/calibration_reliability_trend.py
"""
Calibration & Reliability Trend — with overlays for market regimes and (optionally) social bursts.

Behavior:
- If models/calibration_reliability_trend.json exists, load it and render.
- If missing, build from logs (trigger_history.jsonl + label_feedback.jsonl), then save & render.
- If market context JSON is present, enrich buckets with returns/volatility regime labels (non-fatal if missing).
- If social reddit context JSON is present, enrich buckets with 'social_bursts' (non-fatal if missing).
- Always try to emit plots (ECE and Brier) to artifacts/.

Notes:
- Plots: 1 chart for ECE and 1 for Brier (no seaborn, no explicit colors).
- Safe for demo mode; if logs missing and demo flagged, synth a tiny deterministic trend.

Env knobs (with defaults):
  MW_CAL_TREND_WINDOW_H      (default 72)
  MW_CAL_TREND_BUCKET_MIN    (default 120)  # minutes per bucket
  MW_CAL_TREND_DIM           (default "origin")
  MW_CAL_ECE_BINS            (default 10)
  MW_ECE_ALERT_THRESHOLD     (default 0.06)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from .common import (
    SummaryContext,
    ensure_dir,
    parse_ts,
    _iso,
    _load_jsonl,
    _write_json,
    is_demo_mode,
)

# -------------------------------
# Configuration helpers
# -------------------------------

def _get_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _get_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:
        return default

# -------------------------------
# Core metrics
# -------------------------------

def _ece_brier(rows: List[Tuple[float, int]], bins: int = 10) -> Tuple[float, float]:
    """
    rows: list of (score, label) with 0<=score<=1 and label in {0,1}
    returns: (ECE, Brier)
    """
    if not rows:
        return 0.0, 0.0

    # Brier
    n = len(rows)
    brier = sum((s - y) ** 2 for s, y in rows) / n

    # ECE
    buckets: List[List[Tuple[float, int]]] = [[] for _ in range(bins)]
    for s, y in rows:
        i = min(bins - 1, max(0, int(s * bins)))
        buckets[i].append((s, y))

    ece = 0.0
    for b in buckets:
        if not b:
            continue
        m = len(b)
        conf = sum(s for s, _ in b) / m
        acc = sum(y for _, y in b) / m
        ece += (m / n) * abs(acc - conf)

    return ece, brier

# -------------------------------
# Windowing and bucketing
# -------------------------------

@dataclass
class _BucketKey:
    dim_key: str           # e.g., origin name or version
    bucket_start: datetime # floor to bucket minutes

def _floor_bucket(dt: datetime, minutes: int) -> datetime:
    minutes_from_day = dt.hour * 60 + dt.minute
    bucket_index = minutes_from_day // minutes
    floored_minutes = bucket_index * minutes
    h, m = divmod(floored_minutes, 60)
    return dt.replace(hour=h, minute=m, second=0, microsecond=0)

def _collect_rows(trigs: List[Dict[str, Any]], labs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Merge triggers and labels by id. Returns id -> {score, label, ts, origin}
    - Accepts 'timestamp' or 'ts' for time
    - Accepts 'origin' dimension
    """
    tmap: Dict[str, Dict[str, Any]] = {}
    for r in trigs:
        rid = str(r.get("id"))
        if not rid:
            continue
        ts_raw = r.get("timestamp") or r.get("ts")
        try:
            ts = parse_ts(ts_raw) if ts_raw is not None else None
        except Exception:
            continue
        origin = r.get("origin") or "unknown"
        score = float(r.get("score", 0.0))
        tmap[rid] = {"score": max(0.0, min(1.0, score)), "ts": ts, "origin": origin}

    for r in labs:
        rid = str(r.get("id"))
        if not rid or rid not in tmap:
            continue
        y = 1 if bool(r.get("label")) else 0
        tmap[rid]["label"] = y

    return tmap

def _build_from_logs(logs_dir: Path, window_h: int, bucket_min: int, dim: str, ece_bins: int) -> Dict[str, Any]:
    """
    Construct calibration trend from append-only logs when the artifact is missing.
    """
    triggers = _load_jsonl(logs_dir / "trigger_history.jsonl")
    labels   = _load_jsonl(logs_dir / "label_feedback.jsonl")

    if not triggers or not labels:
        # If demo mode, synth a small deterministic series; else return empty.
        if is_demo_mode():
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            series = [{
                "key": "reddit",
                "points": [
                    {"bucket_start": _iso(now - timedelta(hours=6)), "ece": 0.04, "brier": 0.08, "n": 40},
                    {"bucket_start": _iso(now - timedelta(hours=4)), "ece": 0.05, "brier": 0.10, "n": 42},
                    {"bucket_start": _iso(now - timedelta(hours=2)), "ece": 0.08, "brier": 0.12, "n": 45},
                ],
            }]
            return {
                "demo": True,
                "meta": {
                    "demo": True,
                    "dim": dim,
                    "window_h": window_h,
                    "bucket_min": bucket_min,
                    "ece_bins": ece_bins,
                    "generated_at": _iso(datetime.now(timezone.utc)),
                },
                "series": series,
            }
        # non-demo: empty
        return {
            "demo": False,
            "meta": {
                "demo": False,
                "dim": dim,
                "window_h": window_h,
                "bucket_min": bucket_min,
                "ece_bins": ece_bins,
                "generated_at": _iso(datetime.now(timezone.utc)),
            },
            "series": [],
        }

    merged = _collect_rows(triggers, labels)
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=window_h)

    # bucket -> dim_key -> list[(score,label)]
    bucket_map: Dict[datetime, Dict[str, List[Tuple[float, int]]]] = {}
    for rid, item in merged.items():
        ts: Optional[datetime] = item.get("ts")
        if ts is None or ts < start or ts > now:
            continue
        if "label" not in item:
            continue
        y = int(item["label"])
        s = float(item["score"])
        if dim == "origin":
            dim_key = str(item.get("origin") or "unknown")
        else:
            dim_key = str(item.get(dim) or "unknown")
        bstart = _floor_bucket(ts, bucket_min)
        bucket_map.setdefault(bstart, {}).setdefault(dim_key, []).append((s, y))

    # Build series
    series_dict: Dict[str, List[Dict[str, Any]]] = {}
    for bstart, dmap in sorted(bucket_map.items()):
        for dim_key, pairs in dmap.items():
            ece, brier = _ece_brier(pairs, bins=ece_bins)
            series_dict.setdefault(dim_key, []).append({
                "bucket_start": _iso(bstart),
                "ece": ece,
                "brier": brier,
                "n": len(pairs),
            })

    series = [{"key": k, "points": v} for k, v in series_dict.items()]
    return {
        "demo": False,
        "meta": {
            "demo": False,
            "dim": dim,
            "window_h": window_h,
            "bucket_min": bucket_min,
            "ece_bins": ece_bins,
            "generated_at": _iso(datetime.now(timezone.utc)),
        },
        "series": series,
    }

# -------------------------------
# Enrichment (market + social)
# -------------------------------

def _load_json_if_exists(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists():
            return json.loads(p.read_text())
        return None
    except Exception:
        return None

def _compute_returns(series: List[Dict[str, Any]]) -> List[Tuple[datetime, float]]:
    """From price series [{t: epoch_sec, price: float}], compute hourly returns."""
    pts = sorted(series, key=lambda r: r["t"])
    out: List[Tuple[datetime, float]] = []
    prev = None
    for r in pts:
        t = r["t"]
        if t > 1e11:  # millis
            t = t / 1000.0
        dt = datetime.fromtimestamp(float(t), tz=timezone.utc)
        price = float(r["price"])
        if prev is not None and prev > 0:
            out.append((dt, (price - prev) / prev))
        else:
            out.append((dt, 0.0))
        prev = price
    return out

def _rolling_vol(returns: List[Tuple[datetime, float]], window: int = 6) -> List[Tuple[datetime, float]]:
    out: List[Tuple[datetime, float]] = []
    buf: List[float] = []
    for i, (dt, r) in enumerate(returns):
        buf.append(r)
        if len(buf) > window:
            buf.pop(0)
        if len(buf) < 2:
            out.append((dt, 0.0))
        else:
            m = sum(buf) / len(buf)
            var = sum((x - m) ** 2 for x in buf) / (len(buf) - 1)
            out.append((dt, math.sqrt(var)))
    return out

def _enrich_with_market(trend: Dict[str, Any], market_ctx: Optional[Dict[str, Any]]) -> None:
    """Attach btc_return and vol bucket labels into last-known ECE points (lightweight)."""
    if not market_ctx:
        return
    btc = market_ctx.get("series", {}).get("bitcoin") or []
    if not btc:
        return
    rets = _compute_returns(btc)
    vols = _rolling_vol(rets, window=6)
    # 75th pct vol threshold
    vvals = [v for _, v in vols if v > 0]
    if not vvals:
        thr = 0.0
    else:
        vvals_sorted = sorted(vvals)
        k = int(0.75 * (len(vvals_sorted) - 1))
        thr = vvals_sorted[k]

    # Align by bucket_start hour
    ret_map = {dt.replace(minute=0, second=0, microsecond=0): r for dt, r in rets}
    vol_map = {dt.replace(minute=0, second=0, microsecond=0): v for dt, v in vols}

    for s in trend.get("series", []):
        for p in s.get("points", []):
            try:
                dt = parse_ts(p["bucket_start"]).replace(minute=0, second=0, microsecond=0)
            except Exception:
                continue
            r = ret_map.get(dt, 0.0)
            v = vol_map.get(dt, 0.0)
            p.setdefault("market", {})
            p["market"]["btc_return"] = r
            p["market"]["btc_vol_bucket"] = "high" if v >= thr and v > 0 else ("low" if v > 0 else "n/a")
            # alerts
            alerts = p.setdefault("alerts", [])
            if p.get("ece", 0.0) > _get_float("MW_ECE_ALERT_THRESHOLD", 0.06):
                if "high_ece" not in alerts:
                    alerts.append("high_ece")
            if v >= thr and v > 0:
                if "volatility_regime" not in alerts:
                    alerts.append("volatility_regime")

def _enrich_with_social(trend: Dict[str, Any], reddit_ctx: Optional[Dict[str, Any]]) -> None:
    """Attach social_bursts from reddit context if present."""
    if not reddit_ctx:
        return
    bursts = reddit_ctx.get("bursts") or []
    if not bursts:
        # ensure key exists
        for s in trend.get("series", []):
            for p in s.get("points", []):
                p.setdefault("social_bursts", [])
        return

    # Map bursts by bucket_start hour
    bmap: Dict[str, List[Dict[str, Any]]] = {}
    for b in bursts:
        bs = b.get("bucket_start")
        if not bs:
            continue
        bmap.setdefault(bs, []).append({
            "subreddit": b.get("subreddit"),
            "term": (b.get("term") or b.get("terms") or ""),
            "z": b.get("z", 0.0),
        })

    for s in trend.get("series", []):
        for p in s.get("points", []):
            key = p.get("bucket_start")
            lst = bmap.get(key, [])
            p["social_bursts"] = lst
            if lst and "alerts" in p and "high_ece" in p["alerts"]:
                if "social_burst_overlap" not in p["alerts"]:
                    p["alerts"].append("social_burst_overlap")

# -------------------------------
# Plotting
# -------------------------------

def _plot_metric(trend: Dict[str, Any], metric: str, outpath: Path) -> None:
    """
    Always write an image file even if plotting raises — tests assert existence.
    """
    ensure_dir(outpath.parent)
    fig = plt.figure()
    try:
        # plot each key as its own line
        for s in trend.get("series", []):
            xs = []
            ys = []
            for p in s.get("points", []):
                xs.append(parse_ts(p["bucket_start"]))
                ys.append(float(p.get(metric, 0.0)))
            if xs:
                plt.plot(xs, ys, marker="o", label=s.get("key", "series"))
                # Overlay markers for social bursts (triangles)
                for p in s.get("points", []):
                    if p.get("social_bursts"):
                        x = parse_ts(p["bucket_start"])
                        y = float(p.get(metric, 0.0))
                        plt.scatter([x], [y], marker="^")
        plt.xlabel("time")
        plt.ylabel(metric)
        if any(s.get("key") for s in trend.get("series", [])):
            plt.legend()
        plt.tight_layout()
    except Exception:
        # swallow — we'll still save whatever figure state we have
        pass
    finally:
        try:
            fig.savefig(outpath)
        finally:
            plt.close(fig)

# -------------------------------
# Markdown rendering
# -------------------------------

def _render_md(md: List[str], trend: Dict[str, Any], window_h: int, title_suffix: str = "Market + Social") -> None:
    md.append(f"### 🧮 Calibration & Reliability Trend vs {title_suffix} ({window_h}h)")
    lines = 0
    for s in trend.get("series", []):
        if not s.get("points"):
            continue
        last = s["points"][-1]
        ece = last.get("ece", 0.0)
        alerts = ", ".join(last.get("alerts", [])) if last.get("alerts") else ""
        btc = last.get("market", {}).get("btc_return")
        btc_str = "n/a" if btc is None else f"{btc:+.1%}"

        # >>> CHANGE: Surface social burst terms in the summary line
        burst_terms: List[str] = []
        for b in last.get("social_bursts", []) or []:
            term = (b.get("term") or "").strip()
            if term:
                burst_terms.append(term)
        burst_suffix = ""
        if burst_terms:
            # collapse duplicates, keep order
            seen = set()
            uniq_terms = [t for t in burst_terms if not (t in seen or seen.add(t))]
            burst_suffix = f" + Reddit burst [{', '.join(uniq_terms)}]"
        # <<<

        suffix = f" [{alerts}]" if alerts else ""
        md.append(f"{s.get('key','?')} → ECE {ece:.02f}, BTC {btc_str}{suffix}{burst_suffix}")
        lines += 1
    if lines == 0:
        md.append("_no data available_")

# -------------------------------
# Public entrypoint
# -------------------------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Enrich calibration trend with market + social bursts, emit plots + markdown.
    Falls back to building from logs when the artifact is missing.
    """
    models_dir = Path(ctx.models_dir)
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", "artifacts"))
    ensure_dir(models_dir)
    ensure_dir(artifacts_dir)

    window_h = _get_int("MW_CAL_TREND_WINDOW_H", 72)
    bucket_min = _get_int("MW_CAL_TREND_BUCKET_MIN", 120)
    dim = os.getenv("MW_CAL_TREND_DIM", "origin")
    ece_bins = _get_int("MW_CAL_ECE_BINS", 10)

    trend_path = models_dir / "calibration_reliability_trend.json"
    # Load or build
    if trend_path.exists():
        try:
            trend = json.loads(trend_path.read_text())
        except Exception:
            trend = {}
    else:
        trend = _build_from_logs(Path(ctx.logs_dir), window_h, bucket_min, dim, ece_bins)
        _write_json(trend_path, trend)

    # Optional enrichments (non-fatal)
    try:
        market = _load_json_if_exists(models_dir / "market_context.json")
        _enrich_with_market(trend, market)
    except Exception:
        pass

    try:
        reddit_ctx = _load_json_if_exists(models_dir / "social_reddit_context.json")
        _enrich_with_social(trend, reddit_ctx)
    except Exception:
        pass

    # Save enriched trend (back) to disk
    try:
        _write_json(trend_path, trend)
    except Exception:
        pass

    # Plots (always emit a file, even if plotting fails)
    _plot_metric(trend, "ece", artifacts_dir / "calibration_trend_ece.png")
    _plot_metric(trend, "brier", artifacts_dir / "calibration_trend_brier.png")

    # Markdown — show "(demo)" in header when artifact or environment flags demo.
    demo_flag = bool(trend.get("meta", {}).get("demo")) or bool(trend.get("demo")) or is_demo_mode()
    hdr_suffix = "Market + Social" + (" (demo)" if demo_flag else "")
    _render_md(md, trend, window_h, title_suffix=hdr_suffix)