# scripts/ml/feature_builder.py
from __future__ import annotations

import os, json
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd

# Import regime detector for market regime features
try:
    from .regime_detector import detect_market_regime, get_regime_config, regime_filtering_enabled
    _REGIME_AVAILABLE = True
except ImportError:
    _REGIME_AVAILABLE = False

# =========================
# Core helpers
# =========================
def _returns(df: pd.DataFrame, col="close") -> pd.Series:
    return df[col].pct_change().fillna(0.0)

def _atr(df: pd.DataFrame, n=14) -> pd.Series:
    # If high/low not present (common in hourly aggregates), approximate
    if not {"high","low","close"}.issubset(df.columns):
        c = df["close"].astype(float)
        r = c.pct_change().abs().fillna(0.0)
        h = c * (1 + r)
        l = c * (1 - r)
    else:
        h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(),
                    (h - prev_c).abs(),
                    (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean().fillna(method="bfill").fillna(0.0)

def _sma_gap(df: pd.DataFrame, col="close", win=24) -> pd.Series:
    sma = df[col].rolling(win, min_periods=max(2, win//2)).mean()
    return ((df[col] - sma) / (sma.replace(0, np.nan))).fillna(0.0)

def _zscore(s: pd.Series) -> pd.Series:
    m, sd = s.mean(), s.std(ddof=1)
    if pd.isna(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd

def _vol(df: pd.DataFrame, ret_col: str, win: int) -> pd.Series:
    r = df[ret_col].astype(float)
    return r.rolling(win, min_periods=max(3, win//3)).std(ddof=1).fillna(0.0)

# =========================
# Social (reddit) helpers
# =========================
def _load_social_hourly(root: Path) -> pd.DataFrame:
    """
    Reads logs/social_reddit.jsonl into hourly counts.
    Expected row keys: {ts|timestamp|time}, subreddit
    """
    p = root / "logs" / "social_reddit.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            j = json.loads(ln)
            ts = j.get("ts") or j.get("timestamp") or j.get("time")
            if not ts:
                continue
            s = str(ts)
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            t = pd.to_datetime(s, utc=True).floor("H")
            rows.append(t)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame({"ts": rows, "n": 1.0})
    return df.groupby("ts", as_index=True)["n"].sum().to_frame()

def _social_score_series(root: Path, smooth_h: int = 6) -> pd.DataFrame:
    """ Simple normalized activity proxy, smoothed. """
    df = _load_social_hourly(root)
    if df.empty:
        return pd.DataFrame(columns=["social_score"])
    rate = df["n"].asfreq("H").fillna(0.0)
    sm = rate.rolling(smooth_h, min_periods=1).mean()
    z = _zscore(sm).clip(-5, 5)
    score = 1.0 / (1.0 + np.exp(-z))
    out = pd.DataFrame({"social_score": score})
    out.index.name = "ts"
    return out

def _social_burst_series(root: Path, z_thresh=2.0, smooth_h=3) -> pd.DataFrame:
    df = _load_social_hourly(root)
    if df.empty:
        return pd.DataFrame(columns=["social_burst"])
    rate = df["n"].asfreq("H").fillna(0.0)
    sm = rate.rolling(smooth_h, min_periods=1).mean()
    z  = _zscore(sm).fillna(0.0)
    out = pd.DataFrame({"social_burst": (z > z_thresh).astype(float)})
    out.index.name = "ts"
    return out

# =========================
# Price burst helper
# =========================
def _price_burst(F: pd.DataFrame, ret_col="r_1h",
                 short_h=6, long_h=48, z_thresh=2.0) -> pd.Series:
    r = F[ret_col].fillna(0.0)
    short = r.rolling(short_h, min_periods=short_h).mean()
    long  = r.rolling(long_h,  min_periods=long_h).mean()
    spread = (short - long).fillna(0.0)
    z = _zscore(spread).fillna(0.0)
    return (z > z_thresh).astype(float)

# =========================
# Public: build_features
# =========================
def build_features(prices_map: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Input:
      prices_map[symbol] → DataFrame with at least:
        ts (datetime-like, tz-aware preferred), close (float)
        optional: high, low
    Output:
      dict[symbol] → feature DataFrame with 'ts' and the feature columns.
    """
    root = Path(".").resolve()
    out: Dict[str, pd.DataFrame] = {}

    # Precompute social series once
    social_score_df = _social_score_series(root)
    social_burst_df = _social_burst_series(
        root,
        z_thresh=float(os.getenv("BURST_Z", "2.0")),
        smooth_h=int(os.getenv("MW_SOC_SMOOTH", "3") or "3")
    )

    burst_short = int(os.getenv("BURST_SHORT_H", "6"))
    burst_long  = int(os.getenv("BURST_LONG_H", "48"))

    for sym, df in prices_map.items():
        df = df.copy()
        # ensure ts indexed and hourly
        if "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            df = df.sort_values("ts").set_index("ts")
        else:
            df = df.sort_index()
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("prices_map DataFrame must have 'ts' column or DatetimeIndex")

        # Core returns
        df["r_1h"] = _returns(df, "close")
        df["r_3h"] = df["close"].pct_change(3).fillna(0.0)
        df["r_6h"] = df["close"].pct_change(6).fillna(0.0)

        # Volatility & ATR
        df["vol_6h"] = _vol(df, "r_1h", 6)
        df["atr_14h"] = _atr(df, 14)

        # SMA gap (normalized)
        df["sma_gap"] = _sma_gap(df, "close", 24)

        # High-vol flag (top decile of vol_6h)
        try:
            q90 = float(df["vol_6h"].quantile(0.90))
            df["high_vol"] = (df["vol_6h"] >= q90).astype(float)
        except Exception:
            df["high_vol"] = 0.0

        # Join social_score + social_burst on index
        F = df.copy()
        if not social_score_df.empty:
            F = F.join(social_score_df, how="left")
        if not social_burst_df.empty:
            F = F.join(social_burst_df, how="left")

        # Fill safely (avoid .fillna on floats)
        if "social_score" in F.columns:
            F["social_score"] = F["social_score"].fillna(0.0)
        else:
            F["social_score"] = 0.0

        if "social_burst" in F.columns:
            F["social_burst"] = F["social_burst"].fillna(0.0)
        else:
            F["social_burst"] = 0.0

        # Price burst
        try:
            F["price_burst"] = _price_burst(
                F, "r_1h", burst_short, burst_long, float(os.getenv("BURST_Z","2.0"))
            )
        except Exception:
            F["price_burst"] = 0.0

        # Regime detection (optional, enabled by MW_REGIME_ENABLED env var)
        if _REGIME_AVAILABLE and str(os.getenv("MW_REGIME_ENABLED", "0")).lower() in {"1", "true", "yes"}:
            try:
                regime_config = get_regime_config()
                # detect_market_regime expects close column in df, with datetime index
                prices_for_regime = df[["close"]].copy()
                regime = detect_market_regime(prices_for_regime, symbol=sym, **regime_config)
                # Add as binary feature (1=trending, 0=choppy)
                F["regime_trending"] = (regime == "trending").astype(float)
            except Exception as e:
                print(f"[WARN] Failed to add regime feature for {sym}: {e}")
                F["regime_trending"] = 0.0
        else:
            F["regime_trending"] = 0.0

        # Safety: ensure all expected features present
        expected = [
            "r_1h","r_3h","r_6h",
            "vol_6h","atr_14h","sma_gap","high_vol",
            "social_score","price_burst","social_burst",
            "regime_trending",  # NEW
        ]
        for c in expected:
            if c not in F.columns:
                F[c] = 0.0

        # keep as regular column 'ts'
        F = F.reset_index().rename(columns={"index":"ts"})
        out[sym] = F

    return out