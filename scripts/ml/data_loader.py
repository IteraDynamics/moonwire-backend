# scripts/ml/data_loader.py
from __future__ import annotations

import os
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Env Switches ----------
# Force demo regardless of cache/network
MW_FORCE_DEMO = os.getenv("MW_FORCE_DEMO", "0") == "1"
# Require real data; raise if demo fallback happens
MW_REQUIRE_REAL = os.getenv("MW_REQUIRE_REAL", "0") == "1"
# Lookback override (used only by callers; here for visibility)
DEFAULT_LOOKBACK_DAYS = int(os.getenv("MW_ML_LOOKBACK_DAYS", "180"))

# ---------- CoinGecko basics ----------
# Simple ID map (extend if you add more symbols)
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}
CG_BASE = "https://api.coingecko.com/api/v3"

# ---------- Helpers ----------
def _parquet_path(symbol: str, timeframe: str = "1h") -> Path:
    return DATA_DIR / f"prices_{symbol.upper()}_{timeframe}.parquet"

def _write_prices_manifest(source: str, symbols: List[str], lookback_days: int, reason: str = "") -> None:
    manifest = {
        "source": source,             # "coingecko" | "demo" | "cache"
        "symbols": [s.upper() for s in symbols],
        "lookback_days": lookback_days,
        "ts_utc": pd.Timestamp.utcnow().isoformat(),
        "reason": reason,
    }
    (MODELS_DIR / "prices_manifest.json").write_text(json.dumps(manifest, indent=2))

def _cache_ok_for_lookback(df: pd.DataFrame, lookback_days: int) -> bool:
    if df.empty:
        return False
    ts_max = pd.to_datetime(df["ts"].iloc[-1], utc=True)
    ts_min = pd.to_datetime(df["ts"].iloc[0], utc=True)
    need = pd.Timestamp.utcnow(tz="UTC") - pd.Timedelta(days=lookback_days)
    # Cache is "ok" if it spans back at least to our needed min time
    return ts_min <= need <= ts_max

def _load_cache(symbol: str) -> Optional[pd.DataFrame]:
    fp = _parquet_path(symbol)
    if not fp.exists():
        return None
    try:
        df = pd.read_parquet(fp)
        # sanitize columns + types
        cols = ["ts", "open", "high", "low", "close", "volume"]
        df = df[cols].copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df
    except Exception:
        return None

def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    fp = _parquet_path(symbol)
    try:
        df.to_parquet(fp, index=False)
    except Exception:
        # Don't fail the pipeline on cache write issues
        pass

def _to_ohlcv_from_cg_rows(rows: List[List[float]]) -> pd.DataFrame:
    """
    CoinGecko market_chart returns minute or hourly price points as [ms, price].
    We don't get OHLCV per hour directly; we approximate:
      - resample to 1H, and use open/high/low/close from the minute-ish series
      - volume is not provided by this endpoint; set to 0.0 (or synthetic)
    """
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])

    ts = pd.to_datetime([r[0] for r in rows], unit="ms", utc=True)
    price = pd.Series([r[1] for r in rows], index=ts, dtype="float64")
    df = price.to_frame("price")
    # resample to hourly bars
    o = df["price"].resample("1H", origin="start").first()
    h = df["price"].resample("1H", origin="start").max()
    l = df["price"].resample("1H", origin="start").min()
    c = df["price"].resample("1H", origin="start").last()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})
    out["volume"] = 0.0  # no volume in this endpoint; leave 0.0 for now
    out = out.dropna()
    out = out.reset_index().rename(columns={"index": "ts"})
    return out[["ts", "open", "high", "low", "close", "volume"]]

def _fetch_from_coingecko(symbol: str, lookback_days: int) -> pd.DataFrame:
    """
    Minimal, dependency-free fetch using the public HTTP API.
    Endpoint: /coins/{id}/market_chart?vs_currency=usd&days={lookback_days}&interval=hourly
    Returns hourly close points; we approximate OHLC via resample on close ticks.
    """
    import requests  # local import to avoid hard dep if environment is minimal
    cg_id = COINGECKO_IDS.get(symbol.upper())
    if not cg_id:
        raise RuntimeError(f"No CoinGecko id mapping for symbol {symbol}")

    # CoinGecko days param accepts integers and 'max'; we clamp to at least 1
    days = max(1, int(lookback_days))
    url = f"{CG_BASE}/coins/{cg_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days), "interval": "hourly"}
    # Gentle retry
    last_err = None
    for _ in range(3):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 200:
                payload = resp.json()
                prices = payload.get("prices", [])  # [ [ts_ms, price], ... ]
                df = _to_ohlcv_from_cg_rows(prices)
                # If sparse, drop NA and keep last N days
                if not df.empty:
                    # Clip to exact lookback window
                    cutoff = pd.Timestamp.utcnow(tz="UTC") - pd.Timedelta(days=lookback_days)
                    df = df[df["ts"] >= cutoff].reset_index(drop=True)
                return df
            last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(f"CoinGecko fetch failed for {symbol}: {last_err}")

def _make_demo_prices(symbol: str, lookback_days: int) -> pd.DataFrame:
    """
    Deterministic pseudo-random walk OHLCV for CI/offline.
    """
    hours = lookback_days * 24 + 6  # a bit extra
    # Seed per symbol for reproducibility
    seed = abs(hash(symbol.upper())) % (2**32)
    rng = np.random.default_rng(seed)
    ts = pd.date_range(
        end=pd.Timestamp.utcnow().floor("H").tz_localize("UTC"),
        periods=hours,
        freq="H",
    )
    # Start level per symbol
    bases = {"BTC": 20000.0, "ETH": 1500.0, "SOL": 50.0}
    base = bases.get(symbol.upper(), 100.0)
    # Random walk in log space
    steps = rng.normal(loc=0.0, scale=0.003, size=hours)
    log_price = np.log(base) + np.cumsum(steps)
    close = np.exp(log_price)
    # Build OHLC from close with small intra-bar noise
    spread = np.maximum(0.0005 * close, 0.01)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + spread * rng.uniform(0.2, 1.0, size=hours)
    low = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.0, size=hours)
    volume = rng.uniform(100.0, 10000.0, size=hours)

    df = pd.DataFrame(
        {
            "ts": ts,
            "open": open_.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": volume.astype(float),
        }
    )
    # Clean any anomalies
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    return df

def _slice_lookback(df: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    if df.empty:
        return df
    cutoff = pd.Timestamp.utcnow(tz="UTC") - pd.Timedelta(days=lookback_days)
    return df[df["ts"] >= cutoff].reset_index(drop=True)

# ---------- Public API ----------
def load_prices(symbols: List[str], lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> Dict[str, pd.DataFrame]:
    """
    Load hourly OHLCV for each symbol.
    Order of resolution:
      1) If MW_FORCE_DEMO=1 -> demo (deterministic) + manifest
      2) Cache hit that already spans lookback -> cache + manifest(source="cache")
      3) Try CoinGecko; on success -> save cache + manifest(source="coingecko")
      4) Fallback demo -> save cache + manifest(source="demo"); if MW_REQUIRE_REAL=1, raise
    Returns dict[symbol] -> DataFrame with columns:
      ts(UTC), open, high, low, close, volume (float)
    """
    symbols = [s.upper() for s in symbols]
    # Storage for final frames + source decision
    out: Dict[str, pd.DataFrame] = {}
    chosen_source = None
    demo_reason = ""

    # 1) Force demo
    if MW_FORCE_DEMO:
        for s in symbols:
            df = _make_demo_prices(s, lookback_days)
            _save_cache(s, df)
            out[s] = _slice_lookback(df, lookback_days)
        _write_prices_manifest("demo", symbols, lookback_days, reason="MW_FORCE_DEMO=1")
        if MW_REQUIRE_REAL:
            raise RuntimeError("MW_REQUIRE_REAL=1 but MW_FORCE_DEMO=1 produced demo data")
        return out

    # 2) Try cache for all symbols
    cache_ok_all = True
    cached_frames: Dict[str, pd.DataFrame] = {}
    for s in symbols:
        cdf = _load_cache(s)
        if cdf is None or not _cache_ok_for_lookback(cdf, lookback_days):
            cache_ok_all = False
            break
        cached_frames[s] = _slice_lookback(cdf, lookback_days)
    if cache_ok_all and cached_frames:
        for s in symbols:
            out[s] = cached_frames[s]
        _write_prices_manifest("cache", symbols, lookback_days, reason="spans lookback")
        return out

    # 3) Try CoinGecko fetch
    fetch_success = True
    fetched: Dict[str, pd.DataFrame] = {}
    last_error = ""
    for s in symbols:
        try:
            df = _fetch_from_coingecko(s, lookback_days)
            if df.empty:
                fetch_success = False
                last_error = "empty dataframe from API"
                break
            _save_cache(s, df)
            fetched[s] = _slice_lookback(df, lookback_days)
        except Exception as e:
            fetch_success = False
            last_error = str(e)
            break

    if fetch_success and fetched:
        for s in symbols:
            out[s] = fetched[s]
        _write_prices_manifest("coingecko", symbols, lookback_days)
        return out

    # 4) Fallback to deterministic demo
    demo_reason = f"coingecko_failed: {last_error or 'unknown'}"
    for s in symbols:
        df = _make_demo_prices(s, lookback_days)
        _save_cache(s, df)
        out[s] = _slice_lookback(df, lookback_days)

    _write_prices_manifest("demo", symbols, lookback_days, reason=demo_reason)
    if MW_REQUIRE_REAL:
        raise RuntimeError(f"MW_REQUIRE_REAL=1 but demo fallback used ({demo_reason})")

    return out

# ---------- Quick self-test ----------
if __name__ == "__main__":
    syms = os.getenv("MW_ML_SYMBOLS", "BTC,ETH,SOL").split(",")
    syms = [s.strip().upper() for s in syms if s.strip()]
    lookback = int(os.getenv("MW_ML_LOOKBACK_DAYS", str(DEFAULT_LOOKBACK_DAYS)))
    dfs = load_prices(syms, lookback_days=lookback)
    for k, v in dfs.items():
        print(k, v.head(3), v.tail(3), sep="\n---\n")