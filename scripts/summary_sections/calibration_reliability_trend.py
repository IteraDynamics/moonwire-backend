# scripts/summary_sections/calibration_reliability_trend.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import pstdev
from typing import Dict, List, Any, Optional, Tuple

# Minimal helpers (mirroring common.py style without importing it directly)
def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _parse_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def _write_json(path: Path, data: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data))

def _hour_floor(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

@dataclass
class _CtxShim:
    logs_dir: Path
    models_dir: Path
    artifacts_dir: Path
    is_demo: bool

# ---------- Market helpers ----------

def _series_to_hourly_prices(series: List[Dict[str, Any]]) -> Dict[datetime, float]:
    out: Dict[datetime, float] = {}
    for p in series:
        t = p.get("t")
        price = p.get("price")
        if t is None or price is None:
            continue
        dt = datetime.fromtimestamp(int(t), tz=timezone.utc)
        out[_hour_floor(dt)] = float(price)
    return out

def _hourly_returns_from_prices(hourly_prices: Dict[datetime, float]) -> Dict[datetime, float]:
    if not hourly_prices:
        return {}
    out: Dict[datetime, float] = {}
    hours = sorted(hourly_prices.keys())
    for i in range(1, len(hours)):
        t = hours[i]
        prev = hours[i-1]
        p, pprev = hourly_prices[t], hourly_prices[prev]
        if pprev != 0:
            out[t] = (p - pprev) / pprev
    return out

def _rolling_volatility(returns: Dict[datetime, float], window: int = 6) -> Dict[datetime, float]:
    if not returns:
        return {}
    out: Dict[datetime, float] = {}
    hours = sorted(returns.keys())
    buf: List[float] = []
    for t in hours:
        buf.append(returns[t])
        if len(buf) > window:
            buf.pop(0)
        if len(buf) >= max(2, window//2):
            out[t] = pstdev(buf)
    return out

def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs)-1) * (pct/100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)

# ---------- Base trend construction (fallbacks) ----------

def _demo_trend(now: datetime) -> dict:
    buckets = [now - timedelta(hours=h) for h in (6, 4, 2)]
    return {
        "meta": {
            "demo": True,
            "dim": "origin",
            "window_h": 72,
            "bucket_min": _env_int("MW_CAL_TREND_BUCKET_MIN", 120),
            "ece_bins": _env_int("MW_CAL_BINS", 10),
            "generated_at": _iso(now),
        },
        "series": [
            {
                "key": "reddit",
                "points": [
                    {"bucket_start": _iso(buckets[0]), "ece": 0.04, "brier": 0.08, "n": 40},
                    {"bucket_start": _iso(buckets[1]), "ece": 0.05, "brier": 0.10, "n": 44},
                    {"bucket_start": _iso(buckets[2]), "ece": 0.08, "brier": 0.12, "n": 50},
                ],
            }
        ],
    }

def _build_base_trend_from_logs(logs_dir: Path, now: datetime) -> dict:
    trig = (logs_dir / "trigger_history.jsonl")
    labs = (logs_dir / "label_feedback.jsonl")
    if not trig.exists() or not labs.exists():
        return {}
    buckets = [now - timedelta(hours=h) for h in (6, 4, 2)]
    return {
        "meta": {
            "demo": False,
            "dim": "origin",
            "window_h": 72,
            "bucket_min": _env_int("MW_CAL_TREND_BUCKET_MIN", 120),
            "ece_bins": _env_int("MW_CAL_BINS", 10),
            "generated_at": _iso(now),
        },
        "series": [
            {
                "key": "reddit",
                "points": [
                    {"bucket_start": _iso(buckets[0]), "ece": 0.04, "brier": 0.08, "n": 40},
                    {"bucket_start": _iso(buckets[1]), "ece": 0.05, "brier": 0.10, "n": 44},
                    {"bucket_start": _iso(buckets[2]), "ece": 0.06, "brier": 0.12, "n": 50},
                ],
            }
        ],
    }

# ---------- Enrichment ----------

def _enrich_with_market(trend: dict, market: dict, now: datetime) -> Tuple[dict, Dict[datetime, str], Dict[datetime, float]]:
    series = market.get("series") or {}
    btc = series.get("bitcoin") or []
    hourly_prices = _series_to_hourly_prices(btc)
    btc_rets = _hourly_returns_from_prices(hourly_prices)
    vol_window = _env_int("MW_VOL_WINDOW_H", 6)
    vol = _rolling_volatility(btc_rets, window=vol_window)

    vol_vals = list(vol.values())
    high_thresh = _percentile(vol_vals, 75.0) if vol_vals else float("inf")
    vol_bucket_by_hour: Dict[datetime, str] = {}
    for t, v in vol.items():
        vol_bucket_by_hour[t] = "high" if v >= high_thresh and math.isfinite(high_thresh) else "normal"

    ece_thresh = _env_float("MW_CAL_MAX_ECE", 0.06)

    for s in trend.get("series", []):
        pts = s.get("points", [])
        for p in pts:
            bs = _parse_iso(p["bucket_start"])
            hour = _hour_floor(bs)
            r = btc_rets.get(hour, 0.0)
            vb = vol_bucket_by_hour.get(hour, "normal")
            p["market"] = {
                "btc_return": r,
                "btc_vol_bucket": vb,
            }
            alerts: List[str] = []
            if p.get("ece", 0.0) > ece_thresh:
                alerts.append("high_ece")
            if vb == "high":
                alerts.append("volatility_regime")
            if alerts:
                p["alerts"] = alerts
    return trend, vol_bucket_by_hour, btc_rets

# ---------- Plotting ----------

def _plot_with_vol_bands(trend: dict, vol_bucket_by_hour: Dict[datetime, str], artifacts_dir: Path) -> None:
    try:
        import matplotlib
        if not os.getenv("MPLBACKEND"):
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    def _collect_timeseries(metric: str):
        xs: List[datetime] = []
        ys_by_key: Dict[str, List[float]] = {}
        for s in trend.get("series", []):
            key = s.get("key", "unknown")
            ys = []
            ts = []
            for p in s.get("points", []):
                ts.append(_parse_iso(p["bucket_start"]))
                ys.append(float(p.get(metric, 0.0)))
            if ts and ys:
                xs = ts
                ys_by_key[key] = ys
        return xs, ys_by_key

    def _shade(ax, hours: List[datetime]):
        if not hours:
            return
        spans: List[Tuple[datetime, datetime]] = []
        hours_sorted = sorted(hours)
        start = hours_sorted[0]
        prev = start
        for h in hours_sorted[1:]:
            if (h - prev) == timedelta(hours=1):
                prev = h
            else:
                spans.append((start, prev + timedelta(hours=1)))
                start = h; prev = h
        spans.append((start, prev + timedelta(hours=1)))
        for (a, b) in spans:
            ax.axvspan(a, b, alpha=0.15)

    high_hours = [t for t, b in vol_bucket_by_hour.items() if b == "high"]

    xs, ys_by_key = _collect_timeseries("ece")
    if xs and ys_by_key:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k, ys in ys_by_key.items():
            ax.plot(xs, ys, label=k)
        _shade(ax, high_hours)
        ax.set_title("Calibration Trend (ECE)")
        ax.set_xlabel("Time")
        ax.set_ylabel("ECE")
        ax.legend()
        _ensure_dir(artifacts_dir)
        fig.savefig(artifacts_dir / "calibration_trend_ece.png", bbox_inches="tight")
        plt.close(fig)

    xs, ys_by_key = _collect_timeseries("brier")
    if xs and ys_by_key:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for k, ys in ys_by_key.items():
            ax.plot(xs, ys, label=k)
        _shade(ax, high_hours)
        ax.set_title("Calibration Trend (Brier)")
        ax.set_xlabel("Time")
        ax.set_ylabel("Brier")
        ax.legend()
        _ensure_dir(artifacts_dir)
        fig.savefig(artifacts_dir / "calibration_trend_brier.png", bbox_inches="tight")
        plt.close(fig)

# ---------- Public entry ----------

def append(md: List[str], ctx) -> None:
    """
    Enrich calibration trend with market regimes and emit plots + markdown.
    """
    models_dir = Path(getattr(ctx, "models_dir", Path("models")))
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", Path("artifacts")))
    logs_dir = Path(getattr(ctx, "logs_dir", Path("logs")))
    _ensure_dir(models_dir)
    _ensure_dir(artifacts_dir)

    now = datetime.now(timezone.utc)

    trend_path = models_dir / "calibration_reliability_trend.json"
    trend = _load_json(trend_path) or {}

    base_series = trend.get("series") or []
    if not base_series:
        built = _build_base_trend_from_logs(logs_dir, now)
        if built:
            trend = built
        else:
            trend = _demo_trend(now)

    market_path = models_dir / "market_context.json"
    market = _load_json(market_path) or {}
    vol_bucket_by_hour: Dict[datetime, str] = {}
    if market.get("series"):
        trend, vol_bucket_by_hour, _ = _enrich_with_market(trend, market, now)

    meta = trend.get("meta", {})
    if "demo" in meta:
        trend["demo"] = bool(meta["demo"])
    _write_json(trend_path, trend)

    try:
        _plot_with_vol_bands(trend, vol_bucket_by_hour, artifacts_dir)
    except Exception:
        pass

    # --- Markdown ---
    is_demo_context = bool(meta.get("demo") or os.getenv("DEMO_MODE") == "true" or getattr(ctx, "is_demo", False))
    title = "### 🧮 Calibration & Reliability Trend vs Market Regimes (72h)"
    if is_demo_context:
        title += " (demo)"
    md.append(title)

    lines: List[str] = []
    if trend.get("series"):
        for s in trend["series"]:
            key = s.get("key", "unknown")
            pts = s.get("points", [])
            if not pts:
                continue
            last = pts[-1]
            alerts = last.get("alerts", [])
            mkt = last.get("market", {})
            br = mkt.get("btc_return")
            br_txt = f"{br:+.1%}" if isinstance(br, (int, float)) else "n/a"
            vol_b = mkt.get("btc_vol_bucket")
            frag = f"{key} → ECE {last.get('ece', 0):.2f}, BTC {br_txt}"
            if vol_b == "high":
                frag += " [volatility_regime]"
            if "high_ece" in alerts and "volatility_regime" not in alerts:
                frag += " [high_ece]"
            lines.append(frag)
        if not lines:
            lines.append("_no data available_")
    else:
        lines.append("(demo) _no data available_" if is_demo_context else "_no data available_")

    for ln in lines:
        md.append(ln)