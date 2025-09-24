# scripts/summary_sections/calibration_reliability_trend.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# matplotlib for non-interactive envs (CI)
import matplotlib
matplotlib.use(os.getenv("MPLBACKEND", "Agg"))  # respect env, default Agg
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def _parse_iso(ts: str) -> datetime:
    # handle ...Z or full offset
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _rolling_stdev(vals: List[float], window: int) -> List[float]:
    out: List[float] = []
    q: List[float] = []
    s = 0.0
    ss = 0.0
    for v in vals:
        q.append(v)
        s += v
        ss += v * v
        if len(q) > window:
            old = q.pop(0)
            s -= old
            ss -= old * old
        n = len(q)
        if n >= 2:
            mean = s / n
            var = max(ss / n - mean * mean, 0.0)
            out.append(math.sqrt(var))
        else:
            out.append(0.0)
    return out


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Core enrichment logic
# --------------------------------------------------------------------------------------
def _compute_hourly_returns(series: List[Dict[str, float]]) -> Tuple[List[int], List[float]]:
    """
    Given a list of {"t": epoch_sec, "price": float} ordered oldest->newest,
    compute hourly log returns aligned to each t (newest length equals series length).
    """
    ts: List[int] = [int(p["t"]) for p in series]
    prices: List[float] = [float(p["price"]) for p in series]
    rets: List[float] = [0.0]
    for i in range(1, len(prices)):
        p0, p1 = prices[i - 1], prices[i]
        if p0 > 0:
            rets.append(math.log(p1 / p0))
        else:
            rets.append(0.0)
    return ts, rets


def _align_to_hour_floor(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def _bucketize_by_hour(start: datetime, end: datetime, step_h: int) -> List[datetime]:
    cur = _align_to_hour_floor(start)
    out = []
    while cur <= end:
        out.append(cur)
        cur += timedelta(hours=step_h)
    return out


def _epoch_to_dt(t: int) -> datetime:
    return datetime.fromtimestamp(int(t), tz=timezone.utc)


def _market_regimes_from_context(mc: Dict) -> Dict[str, Dict[str, List]]:
    """
    Returns per-coin dict:
      { coin: { "t": [datetime], "returns": [float], "vol6": [float], "vol_thresh": float, "vol_bucket": ["low"/"med"/"high"] } }
    """
    coins = mc.get("coins", [])
    series = mc.get("series", {})
    out: Dict[str, Dict[str, List]] = {}

    for coin in coins:
        arr = series.get(coin, [])
        if not arr:
            continue
        # normalize to epoch int/float price (handle either {t,price} or [t,price] flavors)
        norm: List[Dict[str, float]] = []
        for p in arr:
            if isinstance(p, dict):
                norm.append({"t": int(p["t"]), "price": float(p["price"])})
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                norm.append({"t": int(p[0]), "price": float(p[1])})
        norm.sort(key=lambda x: x["t"])
        ts, rets = _compute_hourly_returns(norm)
        vol6 = _rolling_stdev(rets, window=6)
        # 75th percentile cut of last 72h (or all)
        lookback_n = min(len(vol6), mc.get("window_hours", 72))
        ref = vol6[-lookback_n:] if lookback_n > 0 else vol6
        thresh = _percentile(ref, 0.75) if ref else float("nan")
        buckets = []
        for v in vol6:
            if math.isnan(thresh):
                buckets.append("unknown")
            elif v >= thresh:
                buckets.append("high")
            elif v >= 0.5 * thresh:
                buckets.append("med")
            else:
                buckets.append("low")
        out[coin] = {
            "t": [ _epoch_to_dt(t) for t in ts ],
            "returns": rets,
            "vol6": vol6,
            "vol_thresh": thresh,
            "vol_bucket": buckets,
        }
    return out


def _attach_market_to_trend(trend: List[Dict], regimes: Dict[str, Dict], ece_thresh: float = 0.06) -> None:
    """
    Mutates each bucket dict in trend to add:
      - "alerts" list (includes "high_ece" and/or "volatility_regime")
      - "market" subobject with btc_return and btc_vol_bucket (favor BTC; fallback to first coin found)
    """
    # choose BTC if present; otherwise first available coin
    coin = "bitcoin" if "bitcoin" in regimes else (next(iter(regimes.keys())) if regimes else None)
    if not coin:
        # nothing to attach
        for row in trend:
            row.setdefault("alerts", [])
        return

    reg = regimes[coin]
    times: List[datetime] = reg["t"]
    returns: List[float] = reg["returns"]
    vol_bucket: List[str] = reg["vol_bucket"]

    def pick(idx_dt: datetime) -> Tuple[float, str]:
        if not times:
            return 0.0, "unknown"
        # nearest by hour index
        # times is hourly; align by hour-floor
        want = _align_to_hour_floor(idx_dt)
        # binary-ish search on small arrays: linear is fine
        best_i = 0
        best_d = abs((times[0] - want).total_seconds())
        for i in range(1, len(times)):
            d = abs((times[i] - want).total_seconds())
            if d < best_d:
                best_i, best_d = i, d
        return returns[best_i], vol_bucket[best_i]

    for row in trend:
        alerts = row.setdefault("alerts", [])
        # bucket_start may be ISO or datetime-like; handle both
        t_raw = row.get("bucket_start") or row.get("t")
        t_dt = _parse_iso(t_raw) if isinstance(t_raw, str) else t_raw
        r, vb = pick(t_dt)
        row["market"] = {
            "btc_return": round(r, 6),
            "btc_vol_bucket": vb,
        }
        if row.get("ece", 0.0) is not None and row["ece"] > ece_thresh and vb == "high":
            if "high_ece" not in alerts:
                alerts.append("high_ece")
            if "volatility_regime" not in alerts:
                alerts.append("volatility_regime")


# --------------------------------------------------------------------------------------
# Public entry point used by Summary assembly
# --------------------------------------------------------------------------------------
def append(md: List[str], ctx) -> None:
    """
    Enrich calibration trend with market regimes and emit plots + markdown.
    Expected ctx: SummaryContext(logs_dir=Path, models_dir=Path, is_demo=bool, ...)
    """
    # Directories
    models_dir: Path = Path(ctx.models_dir)

    # ---- FIX: tolerate SummaryContext without artifacts_dir (tests) ----
    # Priority:
    #   1) ctx.artifacts_dir if present
    #   2) $ARTIFACTS_DIR
    #   3) <repo_root>/artifacts (if GITHUB_WORKSPACE set)
    #   4) <models_dir>/../artifacts
    artifacts_dir: Path
    if hasattr(ctx, "artifacts_dir") and getattr(ctx, "artifacts_dir") is not None:
        artifacts_dir = Path(getattr(ctx, "artifacts_dir"))
    else:
        artifacts_env = os.getenv("ARTIFACTS_DIR")
        if artifacts_env:
            artifacts_dir = Path(artifacts_env)
        else:
            repo_root = Path(os.getenv("GITHUB_WORKSPACE", models_dir.parent))
            artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Inputs
    trend_path = models_dir / "calibration_trend.json"
    mc_path = models_dir / "market_context.json"

    # Load existing trend (created by other step/section)
    trend_obj = _load_json(trend_path, default={"trend": []})
    trend: List[Dict] = trend_obj.get("trend", [])

    # Load market context (works with live or demo-seeded)
    mc = _load_json(mc_path, default={})
    regimes = _market_regimes_from_context(mc)

    # Attach market features + alerts
    ece_thresh = float(os.getenv("MW_CAL_MAX_ECE", "0.06"))
    _attach_market_to_trend(trend, regimes, ece_thresh=ece_thresh)

    # Save updated JSON back
    trend_obj["trend"] = trend
    _save_json(trend_path, trend_obj)

    # --- Plots with volatility bands ---
    # Build a simple hour-indexed series for ECE/Brier and mark high-volatility spans
    times: List[datetime] = []
    ece_vals: List[float] = []
    brier_vals: List[float] = []
    vol_mask: List[bool] = []

    # pick coin for regime overlay (BTC preferred)
    coin = "bitcoin" if "bitcoin" in regimes else (next(iter(regimes.keys())) if regimes else None)
    reg = regimes.get(coin, None)

    for row in trend:
        t_raw = row.get("bucket_start") or row.get("t")
        t_dt = _parse_iso(t_raw) if isinstance(t_raw, str) else t_raw
        times.append(t_dt)
        ece_vals.append(float(row.get("ece", float("nan"))))
        brier_vals.append(float(row.get("brier", float("nan"))))
        vb = (row.get("market") or {}).get("btc_vol_bucket", "unknown")
        vol_mask.append(vb == "high")

    def _plot_series(yvals: List[float], title: str, out_path: Path):
        if not times or not yvals:
            return
        plt.figure(figsize=(10, 4))
        plt.plot(times, yvals)
        # shade high-vol periods
        if any(vol_mask):
            # group consecutive True spans
            start = None
            for i, is_high in enumerate(vol_mask):
                if is_high and start is None:
                    start = times[i]
                if (not is_high or i == len(vol_mask) - 1) and start is not None:
                    end = times[i] if not is_high else times[i]
                    plt.axvspan(start, end, alpha=0.15)  # default color, light shading
                    start = None
        plt.title(title)
        plt.xlabel("time (UTC)")
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()

    _plot_series(ece_vals, "Calibration Trend — ECE (with volatility bands)", artifacts_dir / "calibration_trend_ece.png")
    _plot_series(brier_vals, "Calibration Trend — Brier (with volatility bands)", artifacts_dir / "calibration_trend_brier.png")

    # --- Markdown summary block ---
    # Summarize per origin/version where applicable (tests mostly check presence/format)
    # We’ll roll up simple lines highlighting overlaps.
    lines: List[str] = []
    lines.append("### 🧮 Calibration & Reliability Trend vs Market Regimes (72h)\n")

    # simple heuristics for a few lines
    # group by origin if available
    by_origin: Dict[str, List[Dict]] = {}
    for row in trend:
        origin = row.get("origin") or "overall"
        by_origin.setdefault(origin, []).append(row)

    def _line_for(k: str, rows: List[Dict]) -> str:
        high_ece = [r for r in rows if "high_ece" in r.get("alerts", [])]
        if high_ece:
            return f"{k:10s} → ECE ↑ during high-vol windows [volatility_regime]"
        return f"{k:10s} → stable calibration"

    for origin, rows in sorted(by_origin.items()):
        lines.append(_line_for(origin, rows))

    md.extend(lines)