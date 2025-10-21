# scripts/ml/data_loader.py
from __future__ import annotations

import io
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests


# ---- Paths -----------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _trim_to_fixed_window(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """
    If MW_FIX_END_TS is set (ISO8601), pin the window to [end - lookback_days, end].
    This runs after all other post-processing to ensure a final clean cut.
    """
    end_env = os.getenv("MW_FIX_END_TS", "").strip()
    if not end_env:
        return df
    end = pd.to_datetime(end_env, utc=True)
    start = end - pd.Timedelta(days=int(lookback_days))

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    out = df.loc[(df["ts"] >= start) & (df["ts"] <= end)].sort_values("ts").reset_index(drop=True)
    if len(out) != len(df):
        print(f"[data_loader] Trimmed {len(df) - len(out)} rows outside {start} → {end}")
    return out


# ---- Public API ------------------------------------------------------------

def load_prices(symbols: List[str], lookback_days: int = 180) -> Dict[str, pd.DataFrame]:
    """
    Load hourly OHLCV for the requested symbols.
    Order of preference:
      1) Fresh local cache (parquet)
      2) Fetch from CoinGecko and refresh cache
      3) Demo synthetic prices (deterministic seed per symbol) and cache

    Returns dict[symbol] -> DataFrame with columns:
      ts (UTC tz-aware), open, high, low, close, volume
    """
    out: Dict[str, pd.DataFrame] = {}
    symbols = [s.upper() for s in symbols]

    for s in symbols:
        try:
            df = _load_from_cache(s, lookback_days)
            if df is None:
                df = _fetch_from_coingecko_or_demo(s, lookback_days)
            df = _slice_lookback(df, lookback_days)
            df = _finalize_schema(df)
            df = _trim_to_fixed_window(df, lookback_days)
            out[s] = df
        except Exception as e:
            # hard fallback to demo if anything unexpected happens
            df = _make_demo_prices(s, lookback_days)
            df = _finalize_schema(df)
            out[s] = df
    return out


# ---- Cache helpers ---------------------------------------------------------

def _cache_path(symbol: str) -> Path:
    return DATA_DIR / f"prices_{symbol.upper()}_1h.parquet"


def _load_from_cache(symbol: str, lookback_days: int) -> pd.DataFrame | None:
    p = _cache_path(symbol)
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        # ensure UTC tz-aware
        if df["ts"].dtype.tz is None:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        # if cache covers requested lookback, use it; otherwise treat as stale
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days + 1)
        if df["ts"].min() <= cutoff:
            return df
        # Cache doesn't go far back enough; let fetcher refresh it
        return None
    except Exception:
        return None


def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_cache_path(symbol), index=False)
    except Exception:
        # best effort; ignore cache write errors in CI/demo
        pass


# ---- External fetch (CoinGecko) -------------------------------------------

# Minimal symbol -> CoinGecko ID map (extend as needed)
_COINGECKO_ID = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}

def _fetch_from_coingecko_or_demo(symbol: str, lookback_days: int) -> pd.DataFrame:
    """Try CoinGecko; on any failure fall back to demo and still cache."""
    try:
        df = _fetch_from_coingecko(symbol, lookback_days)
        if df is None or df.empty:
            raise RuntimeError("empty coingecko df")
        _save_cache(symbol, df)
        return df
    except Exception:
        df = _make_demo_prices(symbol, lookback_days)
        _save_cache(symbol, df)
        return df


def _fetch_from_coingecko(symbol: str, lookback_days: int) -> pd.DataFrame | None:
    """
    Calls CoinGecko market_chart endpoint:
      /coins/{id}/market_chart?vs_currency=usd&days={n}
    For days <= 90, CG returns ~hourly data; for larger windows it may be coarser.
    We resample to 1h OHLCV.
    """
    cg_id = _COINGECKO_ID.get(symbol.upper())
    if not cg_id:
        return None

    days_param = max(1, int(lookback_days))
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
    params = {"vs_currency": "usd", "days": days_param, "interval": "hourly"}

    # respect CI / rate limits if env disables external hits
    if os.getenv("MW_OFFLINE", "").lower() in ("1", "true", "yes"):
        return None

    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        return None
    payload = resp.json()

    # payload keys: prices, market_caps, total_volumes; each is [ [ms, value], ... ]
    prices = payload.get("prices", [])
    volumes = payload.get("total_volumes", [])

    if not prices:
        return None

    # Build base close from prices
    ts = [pd.Timestamp(x[0], unit="ms", tz="UTC") for x in prices]
    close = [float(x[1]) for x in prices]
    df = pd.DataFrame({"ts": ts, "close": close})

    # Attach volume if available
    if volumes and len(volumes) == len(prices):
        vol_vals = [float(x[1]) for x in volumes]
        df["volume"] = vol_vals
    else:
        # fallback: small synthetic volume
        df["volume"] = 0.0

    # Resample to exact 1h and reconstruct OHLC from close as a proxy
    df = df.set_index("ts").sort_index()
    # Fill any gaps before OHLC construction
    df = df.asfreq("1H", method="pad")

    o = df["close"].shift(1)  # prev close as open
    h = df["close"].rolling(2).max()
    l = df["close"].rolling(2).min()
    c = df["close"]

    out = pd.DataFrame(
        {
            "ts": df.index,
            "open": o.fillna(c).values,
            "high": h.fillna(c).values,
            "low": l.fillna(c).values,
            "close": c.values,
            "volume": df["volume"].fillna(0.0).values,
        }
    )
    return out.dropna().reset_index(drop=True)


# ---- Demo data (synthetic) -------------------------------------------------

def _make_demo_prices(symbol: str, lookback_days: int) -> pd.DataFrame:
    """
    Deterministic 1h random walk seeded by symbol, with plausible OHLCV.
    """
    seed = (sum(ord(ch) for ch in symbol.upper()) + lookback_days) % (2**32 - 1)
    rng = np.random.default_rng(seed)

    end = pd.Timestamp.now(tz="UTC").floor("H")
    start = end - pd.Timedelta(days=max(lookback_days, 2))  # ensure a bit of warmup
    idx = pd.date_range(start=start, end=end, freq="1H", tz="UTC")

    # random walk on log-returns
    mu = 0.0
    sigma = 0.015  # a bit volatile to generate trades
    rets = rng.normal(mu, sigma, len(idx))
    price0 = 20_000.0 if symbol.upper() == "BTC" else (1_500.0 if symbol.upper() == "ETH" else 50.0)
    close = price0 * np.exp(np.cumsum(rets))
    close = np.maximum(close, 0.01)

    # make OHLC around close
    spread = np.maximum(0.002 * close, 0.01)  # 20 bps-ish band
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum.reduce([open_, close]) + spread
    low = np.minimum.reduce([open_, close]) - spread

    # synthetic volume
    base_vol = 100.0 if symbol.upper() == "BTC" else (80.0 if symbol.upper() == "ETH" else 60.0)
    vol = base_vol * (1.0 + rng.normal(0, 0.25, len(idx)))
    vol = np.maximum(vol, 0.0)

    df = pd.DataFrame(
        {
            "ts": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )
    return df.reset_index(drop=True)


# ---- Post-processing -------------------------------------------------------

def _slice_lookback(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if df.empty:
        return df
    # ensure tz-aware
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    return df[df["ts"] >= cutoff].sort_values("ts").reset_index(drop=True)


def _finalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["ts", "open", "high", "low", "close", "volume"]
    df = df.copy()
    # ensure all columns exist
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[cols].dropna(subset=["ts", "close"]).sort_values("ts").reset_index(drop=True)

    # enforce numeric types
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # coerce to 1H frequency (pad missing, then rebuild OHLC from close if needed)
    # If the input is already 1H uniform, this will be a no-op.
    df = df.set_index("ts").sort_index()
    df = df.asfreq("1H", method="pad")

    # If any of OHLC are missing (due to pad), rebuild minimally from close
    if df[["open", "high", "low"]].isna().any().any():
        c = df["close"]
        o = c.shift(1).fillna(c)
        h = pd.concat([o, c], axis=1).max(axis=1)
        l = pd.concat([o, c], axis=1).min(axis=1)
        df["open"] = df["open"].fillna(o)
        df["high"] = df["high"].fillna(h)
        df["low"] = df["low"].fillna(l)
        df["volume"] = df["volume"].fillna(0.0)

    df = df.reset_index().rename(columns={"index": "ts"})
    return df