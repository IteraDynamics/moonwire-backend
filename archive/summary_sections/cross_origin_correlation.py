# scripts/summary_sections/cross_origin_correlation.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# We use pandas for easy hourly bucketing and alignment
try:
    import pandas as pd
except Exception:  # pragma: no cover - tests should have pandas
    pd = None

from .common import SummaryContext


# -----------------------------
# Helpers: formatting & safety
# -----------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(x, default=None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _pair_lead_text(pair: str, lag: Optional[int]) -> str:
    """
    pair: like "reddit–twitter" or "twitter–market"
    lag: +N means left leads by N hours; -N means right leads by N hours; 0 means synchronous.
    """
    if lag is None:
        return "synchronous"
    if lag == 0:
        return "synchronous"
    left, right = pair.split("–", 1)
    if lag > 0:
        # left leads
        return f"{left} leads by {lag:+d}h"
    else:
        # right leads
        return f"{right} leads by {(-lag):+d}h"


def _strength_tag(r: Optional[float]) -> str:
    if r is None or math.isnan(r):
        return "n/a"
    a = abs(r)
    if a < 0.10:
        return "very weak"
    if a < 0.30:
        return "weak"
    if a < 0.50:
        return "moderate"
    return "strong"


def _fmt_line(pair: str, r: Optional[float], lag: Optional[int]) -> str:
    rtxt = "n/a" if (r is None or (isinstance(r, float) and math.isnan(r))) else f"{r:.2f} ({_strength_tag(r)})"
    ltxt = _pair_lead_text(pair, lag)
    return f"{pair} → r={rtxt} | {ltxt}"


# --------------------------------
# Data loading & aggregation layer
# --------------------------------

@dataclass
class SeriesBundle:
    index: pd.DatetimeIndex
    values: pd.Series  # aligned hourly series


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for cross_origin_correlation section")


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
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


def _hourly_counts_from_reddit(logs_dir: Path, lookback_h: int, now: datetime) -> Optional[SeriesBundle]:
    """Expect logs/social_reddit.jsonl with 'created_utc' timestamps."""
    path = logs_dir / "social_reddit.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return None
    ts = []
    for r in rows:
        t = r.get("created_utc") or r.get("ts_ingested_utc") or r.get("ts")
        try:
            ts.append(pd.to_datetime(t, utc=True))
        except Exception:
            continue
    if not ts:
        return None
    s = pd.Series(1, index=pd.DatetimeIndex(ts))
    s = s.sort_index()
    start = (now - timedelta(hours=lookback_h)).replace(minute=0, second=0, microsecond=0)
    end = now.replace(minute=0, second=0, microsecond=0)
    s = s[(s.index >= start) & (s.index <= end)]
    hourly = s.resample("1H").sum().astype(float)
    idx = pd.date_range(start=start, end=end, freq="1H")
    hourly = hourly.reindex(idx, fill_value=0.0)
    return SeriesBundle(index=hourly.index, values=hourly)


def _hourly_counts_from_twitter(logs_dir: Path, lookback_h: int, now: datetime) -> Optional[SeriesBundle]:
    """Expect logs/social_twitter.jsonl with 'created_utc' timestamps."""
    path = logs_dir / "social_twitter.jsonl"
    rows = _read_jsonl(path)
    if not rows:
        return None
    ts = []
    for r in rows:
        t = r.get("created_utc") or r.get("ts_ingested_utc") or r.get("ts")
        try:
            ts.append(pd.to_datetime(t, utc=True))
        except Exception:
            continue
    if not ts:
        return None
    s = pd.Series(1, index=pd.DatetimeIndex(ts))
    s = s.sort_index()
    start = (now - timedelta(hours=lookback_h)).replace(minute=0, second=0, microsecond=0)
    end = now.replace(minute=0, second=0, microsecond=0)
    s = s[(s.index >= start) & (s.index <= end)]
    hourly = s.resample("1H").sum().astype(float)
    idx = pd.date_range(start=start, end=end, freq="1H")
    hourly = hourly.reindex(idx, fill_value=0.0)
    return SeriesBundle(index=hourly.index, values=hourly)


def _hourly_market_returns_from_logs_or_model(logs_dir: Path, models_dir: Path, lookback_h: int, now: datetime) -> Optional[SeriesBundle]:
    """
    Try logs/market_prices.jsonl first (expects objects with 't' or 'ts' and 'price'),
    else fall back to models/market_context.json (CoinGecko series) and compute BTC 1h returns.
    """
    # 1) Try logs
    log_path = logs_dir / "market_prices.jsonl"
    rows = _read_jsonl(log_path)
    if rows:
        pts = []
        for r in rows:
            t = r.get("t") or r.get("ts") or r.get("timestamp")
            p = r.get("price")
            tt = None
            try:
                # support epoch seconds
                if isinstance(t, (int, float)) and t > 10_000_000:
                    tt = pd.to_datetime(int(t), unit="s", utc=True)
                else:
                    tt = pd.to_datetime(t, utc=True)
            except Exception:
                continue
            pv = _safe_float(p)
            if pv is None:
                continue
            pts.append((tt, pv))
        if pts:
            pts.sort(key=lambda x: x[0])
            s = pd.Series([p for _, p in pts], index=pd.DatetimeIndex([t for t, _ in pts]))
            s = s.sort_index()
            start = (now - timedelta(hours=lookback_h + 6)).replace(minute=0, second=0, microsecond=0)
            end = now.replace(minute=0, second=0, microsecond=0)
            s = s[(s.index >= start) & (s.index <= end)]
            # hourly reindex then returns
            s = s.resample("1H").last().interpolate(limit_direction="both")
            ret = s.pct_change().fillna(0.0)
            ret = ret[(ret.index >= start) & (ret.index <= end)]
            idx = pd.date_range(start=start, end=end, freq="1H")
            ret = ret.reindex(idx, fill_value=0.0)
            return SeriesBundle(index=ret.index, values=ret.astype(float))

    # 2) Fallback to models/market_context.json (as produced by Market Context section)
    mc_path = models_dir / "market_context.json"
    if mc_path.exists():
        try:
            doc = json.loads(mc_path.read_text())
            series = (doc.get("series") or {}).get("bitcoin") or []
            pts = []
            for row in series:
                t = row.get("t")
                price = row.get("price")
                if t is None or price is None:
                    continue
                try:
                    tt = pd.to_datetime(int(t), unit="s", utc=True)
                    pv = float(price)
                except Exception:
                    continue
                pts.append((tt, pv))
            if pts:
                pts.sort(key=lambda x: x[0])
                s = pd.Series([p for _, p in pts], index=pd.DatetimeIndex([t for t, _ in pts]))
                s = s.sort_index()
                start = (now - timedelta(hours=lookback_h + 6)).replace(minute=0, second=0, microsecond=0)
                end = now.replace(minute=0, second=0, microsecond=0)
                s = s[(s.index >= start) & (s.index <= end)]
                s = s.resample("1H").last().interpolate(limit_direction="both")
                ret = s.pct_change().fillna(0.0)
                ret = ret[(ret.index >= start) & (ret.index <= end)]
                idx = pd.date_range(start=start, end=end, freq="1H")
                ret = ret.reindex(idx, fill_value=0.0)
                return SeriesBundle(index=ret.index, values=ret.astype(float))
        except Exception:
            pass

    return None


# ----------------------------
# Correlation & lead/lag math
# ----------------------------

def _pearson(a: pd.Series, b: pd.Series) -> Optional[float]:
    try:
        if len(a) == 0 or len(b) == 0:
            return None
        aa = a.astype(float)
        bb = b.astype(float)
        if aa.std() == 0 or bb.std() == 0:
            return 0.0
        return float(aa.corr(bb))
    except Exception:
        return None


def _lead_lag_by_xcorr(a: pd.Series, b: pd.Series, max_lag_h: int = 6) -> Optional[int]:
    """
    Find lag (in hours) within [-max_lag_h, +max_lag_h] that maximizes correlation.
    +lag means 'a' leads 'b' by lag hours (a shifted earlier).
    -lag means 'b' leads 'a' by |lag| hours.
    """
    try:
        aa = a.astype(float).to_numpy()
        bb = b.astype(float).to_numpy()
        if aa.size < 3 or bb.size < 3:
            return 0
        best_lag = 0
        best_score = -1.0
        # Normalize to zero mean to make correlation comparable
        aa = (aa - aa.mean()) if aa.std() > 0 else aa
        bb = (bb - bb.mean()) if bb.std() > 0 else bb
        for lag in range(-max_lag_h, max_lag_h + 1):
            if lag == 0:
                x = aa
                y = bb
            elif lag > 0:
                # a leads: compare a[:-lag] with b[lag:]
                x = aa[:-lag]
                y = bb[lag:]
            else:  # lag < 0: b leads by -lag
                x = aa[-lag:]
                y = bb[:lag]  # since lag negative, this slices bb[:-|lag|]
            if len(x) < 3 or len(y) < 3 or len(x) != len(y):
                continue
            sx = x.std()
            sy = y.std()
            if sx == 0 or sy == 0:
                score = 0.0
            else:
                score = float(np.corrcoef(x, y)[0, 1])
            if score > best_score:
                best_score = score
                best_lag = lag
        return int(best_lag)
    except Exception:
        return 0


def _build_heatmap(values: Dict[str, Optional[float]], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Matrix order: reddit, twitter, market
        def v(a, b):
            key = f"{a}_{b}"
            if key in values and values[key] is not None:
                return float(values[key])
            key = f"{b}_{a}"
            if key in values and values[key] is not None:
                return float(values[key])
            return 0.0

        mat = np.array([
            [1.0, v("reddit", "twitter"), v("reddit", "market")],
            [v("twitter", "reddit"), 1.0, v("twitter", "market")],
            [v("market", "reddit"), v("market", "twitter"), 1.0],
        ], dtype=float)

        fig = plt.figure(figsize=(4.5, 4.0), dpi=150)
        ax = fig.add_subplot(111)
        cax = ax.imshow(mat, vmin=-1, vmax=1)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["reddit", "twitter", "market"])
        ax.set_yticklabels(["reddit", "twitter", "market"])
        for (i, j), val in np.ndenumerate(mat):
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=9)
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Pearson correlation")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    except Exception:
        # Silent fail on plotting; artifacts are optional
        pass


def _build_leadlag_bars(lags: Dict[str, Optional[int]], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = ["reddit–twitter", "reddit–market", "twitter–market"]
        xs = np.arange(len(labels))
        ys = []
        for lbl in labels:
            if lbl == "reddit–twitter":
                lag = lags.get("reddit_twitter")
            elif lbl == "reddit–market":
                lag = lags.get("reddit_market")
            else:
                lag = lags.get("twitter_market")
            ys.append(0 if lag is None else int(lag))

        fig = plt.figure(figsize=(5.0, 3.2), dpi=150)
        ax = fig.add_subplot(111)
        ax.bar(xs, ys)
        ax.axhline(0, lw=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel("Lag (hours)\n(+ left leads, - right leads)")
        ax.set_title("Lead–Lag (max cross-correlation)")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    except Exception:
        pass


# ---------------
# Main entrypoint
# ---------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Compute/emit:
      - models/cross_origin_correlation.json
      - artifacts/corr_heatmap.png
      - artifacts/corr_leadlag.png
      - Markdown block

    Uses last MW_CORR_LOOKBACK_H (default 72h). In demo mode (MW_DEMO or ctx.is_demo), seeds plausible values.
    """
    _require_pandas()

    lookback_h = int(os.getenv("MW_CORR_LOOKBACK_H", "72") or "72")
    max_lag_h = int(os.getenv("MW_CORR_MAX_LAG_H", "6") or "6")
    demo_mode_env = (os.getenv("MW_DEMO", "").lower() == "true")
    is_demo = bool(getattr(ctx, "is_demo", False)) or demo_mode_env

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Prepare output paths
    models_dir = Path(ctx.models_dir or "models")
    artifacts_dir = Path(ctx.artifacts_dir or "artifacts")
    logs_dir = Path(ctx.logs_dir or "logs")
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    json_out = models_dir / "cross_origin_correlation.json"
    heatmap_png = artifacts_dir / "corr_heatmap.png"
    leadlag_png = artifacts_dir / "corr_leadlag.png"

    # DEMO path: deterministic seeded result
    if is_demo:
        pearson = {
            "reddit_twitter": 0.65,
            "reddit_market": 0.35,
            "twitter_market": 0.40,
        }
        lags = {
            "reddit_twitter": +1,   # reddit leads twitter by +1h
            "reddit_market": +2,    # reddit leads market by +2h
            "twitter_market": 0,    # synchronous
        }
        # Write JSON
        out_doc = {
            "window_hours": lookback_h,
            "generated_at": _utcnow_iso(),
            "pearson": pearson,
            "lead_lag": {
                "reddit→twitter": _pair_lead_text("reddit–twitter", lags["reddit_twitter"]),
                "reddit→market": _pair_lead_text("reddit–market", lags["reddit_market"]),
                "twitter→market": _pair_lead_text("twitter–market", lags["twitter_market"]),
            },
            "demo": True,
        }
        try:
            json_out.write_text(json.dumps(out_doc, indent=2))
        except Exception:
            pass

        # Plots
        try:
            _build_heatmap(pearson, heatmap_png)
            _build_leadlag_bars(lags, leadlag_png)
        except Exception:
            pass

        # Markdown
        md.append(f"\n### 🔗 Cross-Origin Correlations ({lookback_h}h)")
        md.append(_fmt_line("reddit–twitter", pearson["reddit_twitter"], lags["reddit_twitter"]))
        md.append(_fmt_line("reddit–market", pearson["reddit_market"], lags["reddit_market"]))
        md.append(_fmt_line("twitter–market", pearson["twitter_market"], lags["twitter_market"]))
        return

    # Real compute path
    try:
        reddit = _hourly_counts_from_reddit(logs_dir, lookback_h, now)
        twitter = _hourly_counts_from_twitter(logs_dir, lookback_h, now)
        market = _hourly_market_returns_from_logs_or_model(logs_dir, models_dir, lookback_h, now)

        # If any are missing, emit n/a but still produce JSON + markdown
        # Build aligned index if possible
        aligned_index = None
        for b in (reddit, twitter, market):
            if b is not None:
                aligned_index = b.index
                break

        def align_or_none(b: Optional[SeriesBundle]) -> Optional[pd.Series]:
            if b is None:
                return None
            if aligned_index is None:
                return b.values
            return b.values.reindex(aligned_index, fill_value=0.0)

        s_reddit = align_or_none(reddit)
        s_twitter = align_or_none(twitter)
        s_market = align_or_none(market)

        # Compute Pearson r
        r_rt = _pearson(s_reddit, s_twitter) if (s_reddit is not None and s_twitter is not None) else None
        r_rm = _pearson(s_reddit, s_market) if (s_reddit is not None and s_market is not None) else None
        r_tm = _pearson(s_twitter, s_market) if (s_twitter is not None and s_market is not None) else None

        # Lead/lag via cross-correlation
        lag_rt = _lead_lag_by_xcorr(s_reddit, s_twitter, max_lag_h) if (s_reddit is not None and s_twitter is not None) else None
        lag_rm = _lead_lag_by_xcorr(s_reddit, s_market, max_lag_h) if (s_reddit is not None and s_market is not None) else None
        lag_tm = _lead_lag_by_xcorr(s_twitter, s_market, max_lag_h) if (s_twitter is not None and s_market is not None) else None

        # Assemble JSON
        out_doc = {
            "window_hours": lookback_h,
            "generated_at": _utcnow_iso(),
            "pearson": {
                "reddit_twitter": None if r_rt is None else float(r_rt),
                "reddit_market": None if r_rm is None else float(r_rm),
                "twitter_market": None if r_tm is None else float(r_tm),
            },
            "lead_lag": {
                "reddit→twitter": _pair_lead_text("reddit–twitter", lag_rt),
                "reddit→market": _pair_lead_text("reddit–market", lag_rm),
                "twitter→market": _pair_lead_text("twitter–market", lag_tm),
            },
            "demo": False,
        }
        try:
            json_out.write_text(json.dumps(out_doc, indent=2))
        except Exception:
            pass

        # Plots
        try:
            _build_heatmap(
                {
                    "reddit_twitter": r_rt,
                    "reddit_market": r_rm,
                    "twitter_market": r_tm,
                },
                heatmap_png,
            )
            _build_leadlag_bars(
                {
                    "reddit_twitter": lag_rt,
                    "reddit_market": lag_rm,
                    "twitter_market": lag_tm,
                },
                leadlag_png,
            )
        except Exception:
            pass

        # Markdown
        md.append(f"\n### 🔗 Cross-Origin Correlations ({lookback_h}h)")
        md.append(_fmt_line("reddit–twitter", r_rt, lag_rt))
        md.append(_fmt_line("reddit–market", r_rm, lag_rm))
        md.append(_fmt_line("twitter–market", r_tm, lag_tm))

    except Exception as e:
        md.append(f"\n> ❌ Cross-Origin Correlations unavailable: {type(e).__name__}: {e}\n")