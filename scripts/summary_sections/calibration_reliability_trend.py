# scripts/summary_sections/calibration_reliability_trend.py
from __future__ import annotations

import json
import math
import os
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use(os.getenv("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


# ----------------------------
# Utilities
# ----------------------------

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"

def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(ISO_FMT)

def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)

def _safe_load_json(p: Path) -> Optional[Any]:
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        return None
    return None

def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=False))


# ----------------------------
# Config helpers
# ----------------------------

def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


# ----------------------------
# Market context helpers
# ----------------------------

@dataclass
class MarketSeries:
    t: List[datetime]
    price: List[float]

def _series_from_market(series_list: List[Dict[str, Any]]) -> MarketSeries:
    ts = []
    ps = []
    for row in series_list:
        t_raw = row.get("t")
        if isinstance(t_raw, (int, float)):
            # seconds since epoch (most of our seeded data is seconds)
            dt = datetime.fromtimestamp(float(t_raw), tz=timezone.utc)
        else:
            # ISO fallback
            dt = _parse_iso(str(t_raw))
        ts.append(dt)
        ps.append(float(row["price"]))
    return MarketSeries(ts, ps)

def _pct_returns(prices: List[float]) -> List[float]:
    out: List[float] = []
    for i in range(1, len(prices)):
        p0, p1 = prices[i-1], prices[i]
        if p0 == 0:
            out.append(0.0)
        else:
            out.append((p1 - p0) / p0)
    return out

def _rolling_stdev(vals: List[float], window: int) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    if window <= 1:
        return [None]*len(vals)
    for i in range(len(vals)):
        if i + 1 < window:
            out.append(None)
        else:
            window_vals = vals[i+1-window:i+1]
            try:
                out.append(statistics.pstdev(window_vals))
            except Exception:
                out.append(None)
    return out

def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s)-1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1

@dataclass
class MarketFeatures:
    # returns aligned by hour
    btc_rets: Dict[datetime, float]
    eth_rets: Dict[datetime, float]
    sol_rets: Dict[datetime, float]
    # rolling stdev of hourly returns
    btc_vol: Dict[datetime, float]
    eth_vol: Dict[datetime, float]
    sol_vol: Dict[datetime, float]
    # thresholds
    btc_vol_p75: float
    eth_vol_p75: float
    sol_vol_p75: float

def _compute_market_features(market: Dict[str, Any]) -> Optional[MarketFeatures]:
    try:
        series = market.get("series", {})
        btc = _series_from_market(series.get("bitcoin", []))
        if len(btc.price) < 3:
            return None
        # Compute per-coin returns aligned to btc timestamps
        eth = _series_from_market(series.get("ethereum", [])) if "ethereum" in series else MarketSeries([], [])
        sol = _series_from_market(series.get("solana", [])) if "solana" in series else MarketSeries([], [])

        btc_ret_list = _pct_returns(btc.price)
        btc_ret_times = btc.t[1:]

        def _align(to_t: List[datetime], base_times: List[datetime], prices: List[float]) -> Tuple[Dict[datetime, float], Dict[datetime, float]]:
            if not to_t or len(prices) < 2:
                return {}, {}
            # naive positional align: assume same cadence and length; fallback to nearest by index
            # build ret/vol over this coin
            rets = _pct_returns(prices)
            times = to_t[1:]
            # map by timestamp (best effort)
            ret_map: Dict[datetime, float] = {}
            for i, dt in enumerate(times):
                ret_map[dt] = rets[i]
            # rolling vol over last 6 rets
            vol = _rolling_stdev(rets, window=6)
            vol_map: Dict[datetime, float] = {}
            for i, dt in enumerate(times):
                v = vol[i]
                if v is not None:
                    vol_map[dt] = v
            return ret_map, vol_map

        eth_ret_map, eth_vol_map = _align(eth.t, eth.t, eth.price) if eth.t else ({}, {})
        sol_ret_map, sol_vol_map = _align(sol.t, sol.t, sol.price) if sol.t else ({}, {})

        # btc vol
        btc_vol_list = _rolling_stdev(btc_ret_list, window=6)
        btc_vol_map: Dict[datetime, float] = {}
        for i, dt in enumerate(btc_ret_times):
            v = btc_vol_list[i]
            if v is not None:
                btc_vol_map[dt] = v

        # thresholds over available window (ignore Nones)
        btc_vol_vals = [v for v in btc_vol_map.values() if isinstance(v, (int, float))]
        eth_vol_vals = [v for v in eth_vol_map.values() if isinstance(v, (int, float))]
        sol_vol_vals = [v for v in sol_vol_map.values() if isinstance(v, (int, float))]
        btc_p75 = _percentile(btc_vol_vals, 0.75) if btc_vol_vals else 0.0
        eth_p75 = _percentile(eth_vol_vals, 0.75) if eth_vol_vals else 0.0
        sol_p75 = _percentile(sol_vol_vals, 0.75) if sol_vol_vals else 0.0

        # turn lists into dicts keyed by hour for easy lookup
        btc_ret_map: Dict[datetime, float] = {}
        for i, dt in enumerate(btc_ret_times):
            btc_ret_map[dt] = btc_ret_list[i]

        return MarketFeatures(
            btc_rets=btc_ret_map,
            eth_rets=eth_ret_map,
            sol_rets=sol_ret_map,
            btc_vol=btc_vol_map,
            eth_vol=eth_vol_map,
            sol_vol=sol_vol_map,
            btc_vol_p75=btc_p75,
            eth_vol_p75=eth_p75,
            sol_vol_p75=sol_p75,
        )
    except Exception:
        return None

def _market_regime_for_time(mf: MarketFeatures, t: datetime) -> Dict[str, Any]:
    # quantize to hour
    t0 = t.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

    def _bucket(vol_map: Dict[datetime, float], p75: float) -> str:
        v = vol_map.get(t0)
        if v is None:
            return "n/a"
        return "high" if v >= p75 and p75 > 0 else "normal"

    out = {
        "btc_return": mf.btc_rets.get(t0),
        "btc_vol": mf.btc_vol.get(t0),
        "btc_vol_bucket": _bucket(mf.btc_vol, mf.btc_vol_p75),
        "eth_return": mf.eth_rets.get(t0),
        "eth_vol": mf.eth_vol.get(t0),
        "eth_vol_bucket": _bucket(mf.eth_vol, mf.eth_vol_p75) if mf.eth_vol else "n/a",
        "sol_return": mf.sol_rets.get(t0),
        "sol_vol": mf.sol_vol.get(t0),
        "sol_vol_bucket": _bucket(mf.sol_vol, mf.sol_vol_p75) if mf.sol_vol else "n/a",
    }
    return out


# ----------------------------
# Trend computation from logs (fallback when no precomputed)
# ----------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out

def _bucket_floor(ts: datetime, minutes: int) -> datetime:
    q = (ts.minute // minutes) * minutes
    return ts.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=q)

def _ece_simple(scores: List[float], labels: List[int], bins: int = 10) -> float:
    if not scores or not labels or len(scores) != len(labels):
        return 0.0
    bins = max(1, bins)
    # simple equal-width bins on [0,1]
    counts = [0]*bins
    conf_sum = [0.0]*bins
    acc_sum = [0.0]*bins
    for s, y in zip(scores, labels):
        b = min(bins-1, max(0, int(s*bins)))
        counts[b] += 1
        conf_sum[b] += s
        acc_sum[b] += float(y)
    ece = 0.0
    n = len(scores)
    for i in range(bins):
        if counts[i] == 0:
            continue
        avg_c = conf_sum[i] / counts[i]
        avg_a = acc_sum[i] / counts[i]
        ece += (counts[i]/n) * abs(avg_c - avg_a)
    return ece

def _brier(scores: List[float], labels: List[int]) -> float:
    if not scores or not labels or len(scores) != len(labels):
        return 0.0
    return sum((s - y) ** 2 for s, y in zip(scores, labels)) / len(scores)

def _build_trend_from_logs(logs_dir: Path, window_h: int, bucket_min: int, dim: str, ece_bins: int) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(hours=window_h)

    trig = _load_jsonl(logs_dir / "trigger_history.jsonl")
    labs = _load_jsonl(logs_dir / "label_feedback.jsonl")
    if not trig or not labs:
        return {"meta": {"demo": True, "dim": dim, "window_h": window_h, "bucket_min": bucket_min, "ece_bins": ece_bins, "generated_at": _iso(now)},
                "series": []}

    labs_by_id: Dict[str, int] = {}
    for r in labs:
        if "id" in r and "label" in r:
            labs_by_id[r["id"]] = 1 if r["label"] else 0

    # organize by dim + bucket
    buckets: Dict[Tuple[str, datetime], List[Tuple[float, int]]] = {}
    for r in trig:
        try:
            ts = _parse_iso(r["timestamp"])
            if ts < start:
                continue
            key = str(r.get(dim, "unknown"))
            if "score" not in r or "id" not in r:
                continue
            if r["id"] not in labs_by_id:
                continue
            bstart = _bucket_floor(ts, bucket_min)
            buckets.setdefault((key, bstart), []).append((float(r["score"]), labs_by_id[r["id"]]))
        except Exception:
            continue

    # aggregate
    by_series: Dict[str, List[Dict[str, Any]]] = {}
    for (key, bstart), pairs in sorted(buckets.items(), key=lambda x: x[0][1]):
        scores = [s for s, _ in pairs]
        labels = [y for _, y in pairs]
        ece = _ece_simple(scores, labels, bins=ece_bins)
        brier = _brier(scores, labels)
        by_series.setdefault(key, []).append({
            "bucket_start": _iso(bstart),
            "ece": ece,
            "brier": brier,
            "n": len(pairs),
        })

    series_list = [{"key": k, "points": v} for k, v in by_series.items()]
    return {
        "meta": {"demo": False, "dim": dim, "window_h": window_h, "bucket_min": bucket_min, "ece_bins": ece_bins, "generated_at": _iso(now)},
        "series": series_list,
    }


# ----------------------------
# Enrichment with market regimes + plotting + markdown
# ----------------------------

def _enrich_with_market_and_alerts(trend: Dict[str, Any], market: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    # compute market features once
    mf = _compute_market_features(market) if market else None
    max_ece_thresh = _get_env_float("MW_CAL_MAX_ECE", 0.06)

    out = json.loads(json.dumps(trend))  # deep copy
    for s in out.get("series", []):
        new_points = []
        for p in s.get("points", []):
            try:
                t = _parse_iso(p["bucket_start"])
            except Exception:
                t = None
            alerts: List[str] = []
            market_last: Dict[str, Any] = {}
            if mf and t:
                market_last = _market_regime_for_time(mf, t)
                # volatility regime primarily based on BTC
                if market_last.get("btc_vol_bucket") == "high":
                    alerts.append("volatility_regime")
            # calibration alert
            if p.get("ece", 0.0) > max_ece_thresh:
                alerts.append("high_ece")
            # attach
            p2 = dict(p)
            p2["alerts"] = alerts
            if market_last:
                p2["market"] = {
                    "btc_return": market_last.get("btc_return"),
                    "btc_vol_bucket": market_last.get("btc_vol_bucket"),
                    # include ETH/SOL for downstream callouts
                    "eth_return": market_last.get("eth_return"),
                    "eth_vol_bucket": market_last.get("eth_vol_bucket"),
                    "sol_return": market_last.get("sol_return"),
                    "sol_vol_bucket": market_last.get("sol_vol_bucket"),
                }
            new_points.append(p2)
        s["points"] = new_points
    return out

def _plot_with_volatility(trend: Dict[str, Any], market: Optional[Dict[str, Any]], artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    mf = _compute_market_features(market) if market else None

    def _plot(metric: str, fname: str):
        fig, ax = plt.subplots(figsize=(8, 3))
        # gather x spans for BTC high vol
        spans: List[Tuple[datetime, datetime]] = []
        if mf:
            # collect hours where bucket is high
            high_hours = sorted([t for t, v in mf.btc_vol.items() if v >= mf.btc_vol_p75 and mf.btc_vol_p75 > 0])
            # turn discrete hours into contiguous spans
            if high_hours:
                start = high_hours[0]
                prev = high_hours[0]
                for t in high_hours[1:]:
                    if (t - prev) == timedelta(hours=1):
                        prev = t
                    else:
                        spans.append((start, prev + timedelta(hours=1)))
                        start = t
                        prev = t
                spans.append((start, prev + timedelta(hours=1)))

        # shade spans
        for a, b in spans:
            ax.axvspan(a, b, alpha=0.15)

        # lines per series
        for s in trend.get("series", []):
            xs = []
            ys = []
            for p in s.get("points", []):
                try:
                    xs.append(_parse_iso(p["bucket_start"]))
                    ys.append(float(p.get(metric, 0.0)))
                except Exception:
                    pass
            if xs:
                ax.plot(xs, ys, marker="o", label=s.get("key", "series"))

        ax.set_ylabel(metric.upper())
        ax.set_xlabel("time (UTC)")
        ax.grid(True, alpha=0.3)

        # Legend includes a patch entry for high-volatility shading
        vol_handle = Patch(alpha=0.15, label="High volatility (BTC ≥ 75th pct)")
        handles, labels = ax.get_legend_handles_labels()
        handles = list(handles) + [vol_handle]
        labels = list(labels) + ["High volatility (BTC ≥ 75th pct)"]
        ax.legend(handles=handles, labels=labels, loc="upper right")

        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(artifacts_dir / fname, dpi=120)
        plt.close(fig)

    _plot("ece", "calibration_trend_ece.png")
    _plot("brier", "calibration_trend_brier.png")


def append(md: List[str], ctx) -> None:
    """
    Enrich calibration trend with market regimes and emit plots + markdown.
    Behavior:
      - if models/calibration_reliability_trend.json exists: enrich + plot + summarize
      - else: build from logs (or demo seed), then enrich + plot + summarize
    """
    models_dir: Path = Path(getattr(ctx, "models_dir", "models"))
    artifacts_dir: Path = Path(getattr(ctx, "artifacts_dir", "artifacts"))
    logs_dir: Path = Path(getattr(ctx, "logs_dir", "logs"))
    is_demo_ctx: bool = bool(getattr(ctx, "is_demo", False) or os.getenv("DEMO_MODE", "").lower() == "true")

    # load inputs (precomputed trend or build)
    trend_path = models_dir / "calibration_reliability_trend.json"
    market_path = models_dir / "market_context.json"

    base_trend = _safe_load_json(trend_path)
    if not isinstance(base_trend, dict):
        # build from logs or seed demo
        window_h = _get_env_int("MW_CAL_TREND_WINDOW_H", 72)
        bucket_min = _get_env_int("MW_CAL_TREND_BUCKET_MIN", 120)
        dim = os.getenv("MW_CAL_TREND_DIM", "origin")
        ece_bins = _get_env_int("MW_CAL_TREND_ECE_BINS", 10)
        built = _build_trend_from_logs(logs_dir, window_h, bucket_min, dim, ece_bins)
        if (not built.get("series")) and is_demo_ctx:
            # seed simple demo series with 3 buckets for 'reddit'
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            buckets = [now - timedelta(hours=h) for h in (6, 4, 2)]
            built = {
                "meta": {
                    "demo": True,
                    "dim": dim,
                    "window_h": window_h,
                    "bucket_min": bucket_min,
                    "ece_bins": ece_bins,
                    "generated_at": _iso(now),
                },
                "series": [
                    {"key": "reddit", "points": [
                        {"bucket_start": _iso(buckets[0]), "ece": 0.08, "brier": 0.12, "n": 45},
                        {"bucket_start": _iso(buckets[1]), "ece": 0.06, "brier": 0.10, "n": 50},
                        {"bucket_start": _iso(buckets[2]), "ece": 0.09, "brier": 0.13, "n": 52},
                    ]},
                ],
            }
        base_trend = built

    # Header (demo-tag only when actually demo)
    meta = base_trend.get("meta", {}) if isinstance(base_trend, dict) else {}
    header = "### 🧮 Calibration & Reliability Trend vs Market Regimes (72h)"
    if meta.get("demo") or is_demo_ctx:
        header += " (demo)"
    md.append(header)

    # load market (may be absent)
    market = _safe_load_json(market_path)

    # enrich and save + plot
    enriched = _enrich_with_market_and_alerts(base_trend, market)
    # Preserve top-level demo flag (tests may read it either in meta or root)
    if "demo" not in enriched and isinstance(meta.get("demo"), bool):
        enriched["demo"] = meta["demo"]

    _write_json(models_dir / "calibration_reliability_trend.json", enriched)
    _plot_with_volatility(enriched, market, artifacts_dir)

    # markdown summary lines
    def _fmt_pct(x: Optional[float]) -> str:
        if x is None:
            return "n/a"
        return f"{x:+.1%}"

    # Extract the latest market annotation per series point (last point used)
    for s in enriched.get("series", []):
        key = s.get("key", "unknown")
        pts = s.get("points", [])
        if not pts:
            continue
        last = pts[-1]
        ece = last.get("ece")
        ece_disp = f"{ece:.02f}" if isinstance(ece, (int, float)) else "n/a"

        market_last = last.get("market", {})
        btc_ret_disp = _fmt_pct(market_last.get("btc_return"))

        tags = ""
        alerts = last.get("alerts", [])
        if alerts:
            tags = " [" + ",".join(alerts) + "]"

        # main line: keep BTC focus (as before)
        line = f"{key} → ECE {ece_disp}, BTC {btc_ret_disp}{tags}"
        md.append(line)

        # --- ETH/SOL conditional regime callouts (NEW) ---
        # Only add if both high-vol regime and |1h return| ≥ 2%
        eth_r = market_last.get("eth_return")
        sol_r = market_last.get("sol_return")
        eth_vol = market_last.get("eth_vol_bucket")
        sol_vol = market_last.get("sol_vol_bucket")

        eth_callout = (isinstance(eth_r, (int, float)) and abs(eth_r) >= 0.02 and eth_vol == "high")
        sol_callout = (isinstance(sol_r, (int, float)) and abs(sol_r) >= 0.02 and sol_vol == "high")

        if eth_callout:
            md.append(f" ↳ ETH {_fmt_pct(eth_r)} [volatility_regime]")
        if sol_callout:
            md.append(f" ↳ SOL {_fmt_pct(sol_r)} [volatility_regime]")