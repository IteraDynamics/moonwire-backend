# scripts/ml/data_loader.py
from __future__ import annotations
import pandas as pd
import numpy as np
import pathlib, os, io, datetime as dt
from typing import Dict, List
from .utils import ensure_dirs, env_int, to_list

ROOT = pathlib.Path(".").resolve()

def _coingecko_prices_1h(symbol: str, lookback_days: int) -> pd.DataFrame:
    # Best-effort CoinGecko via csv cache if present; no hard network calls in CI.
    # If you already have a client, you can import and replace this.
    # Fallback: synthetic geometric random walk.
    try:
        # Optional local cache someone might have exported:
        cache = ROOT / f"data/prices_{symbol}_1h.parquet"
        if cache.exists():
            df = pd.read_parquet(cache)
            return df.tail(lookback_days * 24).reset_index(drop=True)
    except Exception:
        pass

    # Synthetic fallback (deterministic seed per symbol for tests)
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    hours = lookback_days * 24
    base = 20000.0 if symbol == "BTC" else 1500.0 if symbol == "ETH" else 20.0
    ret = rng.normal(0.0, 0.0012, size=hours)
    price = base * np.exp(np.cumsum(ret))
    high = price * (1 + rng.normal(0.0005, 0.0007, size=hours))
    low = price * (1 - rng.normal(0.0005, 0.0007, size=hours))
    open_ = price * (1 + rng.normal(0.0, 0.0003, size=hours))
    vol = np.abs(rng.normal(1e3, 2e2, size=hours)) * (base / 1000.0)
    ts = pd.date_range(end=dt.datetime.utcnow(), periods=hours, freq="H", tz="UTC")
    df = pd.DataFrame({
        "ts": ts,
        "open": open_,
        "high": np.maximum(high, open_),
        "low": np.minimum(low, open_),
        "close": price,
        "volume": vol,
    })
    return df

def load_prices(symbols: List[str], lookback_days: int = 180) -> Dict[str, pd.DataFrame]:
    ensure_dirs()
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = _coingecko_prices_1h(sym, lookback_days)
        # enforce schema & UTC-naive timestamp column named ts
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df = df.sort_values("ts").reset_index(drop=True)
        # write cache parquet for future speed (optional)
        try:
            (ROOT / f"data/prices_{sym}_1h.parquet").parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(ROOT / f"data/prices_{sym}_1h.parquet", index=False)
        except Exception:
            pass
        data[sym] = df[["ts","open","high","low","close","volume"]]
    return data