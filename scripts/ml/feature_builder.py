# scripts/ml/feature_builder.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Iterable, Tuple

import numpy as np
import pandas as pd


# =========================
# Env flags for social use
# =========================

SOCIAL_ENABLED: bool = str(os.getenv("MW_SOCIAL_ENABLED", "1")).strip().lower() in {
    "1", "true", "yes", "on"
}

_raw_include = os.getenv("MW_SOCIAL_INCLUDE", "").strip()
if _raw_include == "":
    # None => allow social for ALL symbols
    SOCIAL_INCLUDE: Optional[Set[str]] = None
else:
    SOCIAL_INCLUDE = {
        tok.strip().upper()
        for tok in _raw_include.replace(";", ",").split(",")
        if tok.strip()
    }


# =========================
# Helpers
# =========================

def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'ts' datetime64[ns, UTC] column exists and is sorted.
    Accepts either:
      - DataFrame with 'ts' column (datetime-like or epoch-like)
      - DataFrame with DateTimeIndex
    """
    out = df.copy()
    if "ts" in out.columns:
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
    else:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={"index": "ts"})
            out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
        else:
            raise ValueError("Price frame must have a 'ts' column or a DateTimeIndex.")
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return out


def _as_price_df(obj: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    Normalize incoming prices to a DataFrame with ['ts','price'].
    Accepts:
      - DataFrame with 'price' (preferred) or 'close' column
      - Series of prices with DateTimeIndex
    """
    if isinstance(obj, pd.Series):
        df = obj.to_frame(name="price")
    else:
        df = obj.copy()

    if "price" not in df.columns:
        if "close" in df.columns:
            df = df.rename(columns={"close": "price"})
        else:
            # last resort: first numeric column
            num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
            if not num_cols:
                raise ValueError("Could not find a numeric price column.")
            df = df.rename(columns={num_cols[0]: "price"})

    df = _ensure_ts(df)
    # make sure price is float
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])
    return df[["ts", "price"]]


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods)


def _rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window=window, min_periods=max(2, window // 2)).std()


def _atr_like(price: pd.Series, window: int) -> pd.Series:
    """
    ATR proxy when only close/price is available:
      use rolling (max(price) - min(price)) / previous price over the window.
    It isn't classic ATR (needs high/low/close), but is a stable volatility proxy.
    """
    roll_max = price.rolling(window, min_periods=max(2, window // 2)).max()
    roll_min = price.rolling(window, min_periods=max(2, window // 2)).min()
    prev = price.shift(1).replace(0, np.nan)
    atr = (roll_max - roll_min) / prev
    return atr


def _sma_gap(price: pd.Series, window: int) -> pd.Series:
    sma = price.rolling(window=window, min_periods=max(2, window // 2)).mean()
    return (price / sma) - 1.0


def _zscore(x: pd.Series) -> pd.Series:
    m = x.rolling(72, min_periods=12).mean()
    s = x.rolling(72, min_periods=12).std()
    z = (x - m) / s.replace(0, np.nan)
    return z.fillna(0.0)


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # skip bad lines
                continue


def _load_social_series() -> pd.DataFrame:
    """
    Load social activity from logs/social_reddit.jsonl (if present) and return
    an hourly series: ['ts', 'social_count'].
    Very tolerant to schema; uses any of these timestamp fields if found:
    'ts', 'created_utc', 'created_at'.
    """
    log_path = Path("logs/social_reddit.jsonl")
    if not log_path.exists():
        # empty frame with ts for merge_asof
        return pd.DataFrame(columns=["ts", "social_count"], dtype=float)

    rows: list[Tuple[pd.Timestamp, int]] = []
    for rec in _read_jsonl(log_path):
        ts_val = rec.get("ts") or rec.get("created_utc") or rec.get("created_at")
        if ts_val is None:
            continue
        try:
            # support epoch seconds or ISO8601
            if isinstance(ts_val, (int, float)):
                ts = pd.to_datetime(int(ts_val), unit="s", utc=True)
            else:
                ts = pd.to_datetime(ts_val, utc=True)
        except Exception:
            continue
        rows.append((ts, 1))

    if not rows:
        return pd.DataFrame(columns=["ts", "social_count"], dtype=float)

    df = pd.DataFrame(rows, columns=["ts", "one"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # floor to hour and count
    df["ts"] = df["ts"].dt.floor("h")
    s = df.groupby("ts")["one"].sum().rename("social_count").astype(float).reset_index()
    return s[["ts", "social_count"]]


def _build_base_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a price frame ['ts','price'] (hourly), compute base features.
    """
    df = price_df.copy()

    # Returns (assumed hourly bars)
    df["r_1h"] = _pct_change(df["price"], 1)
    df["r_3h"] = _pct_change(df["price"], 3)
    df["r_6h"] = _pct_change(df["price"], 6)

    # Volatility proxies
    df["vol_6h"] = _rolling_vol(df["r_1h"], 6)
    df["atr_14h"] = _atr_like(df["price"], 14)

    # Trend
    df["sma_gap"] = _sma_gap(df["price"], 24)

    # High-vol flag (binary-ish; keep as float)
    med_vol = df["vol_6h"].rolling(96, min_periods=24).median()
    df["high_vol"] = (df["vol_6h"] > (1.5 * med_vol)).astype(float)

    return df


def _merge_social(df: pd.DataFrame, social_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge hourly social counts and convert to a normalized score.
    """
    if social_df.empty:
        df["social_score"] = 0.0
        return df

    # join on hourly ts
    base = df.copy()
    s = social_df.copy()
    base["ts"] = pd.to_datetime(base["ts"], utc=True)
    s["ts"] = pd.to_datetime(s["ts"], utc=True)

    merged = base.merge(s, on="ts", how="left")
    merged["social_count"] = merged["social_count"].fillna(0.0)

    # Normalize -> zscore, then clip to sensible range
    merged["social_score"] = _zscore(merged["social_count"]).clip(-4, 6)
    merged = merged.drop(columns=["social_count"])
    return merged


# =========================
# Public API
# =========================

def build_features(prices: Dict[str, pd.DataFrame | pd.Series]) -> Dict[str, pd.DataFrame]:
    """
    Construct model features for each symbol.

    Input:
      prices: dict like { "BTC": df_or_series, "ETH": ... }
              where each value has at least time + price info.

    Output:
      dict of symbol -> DataFrame with columns:
        ['ts','price','r_1h','r_3h','r_6h','vol_6h','atr_14h','sma_gap','high_vol','social_score']
      (labeling happens downstream)
    """
    out: Dict[str, pd.DataFrame] = {}

    # Load social once (hourly counts), reuse for all merges
    social_hourly = _load_social_series()

    for sym, obj in prices.items():
        sym_u = str(sym).upper()

        # Base price features
        p = _as_price_df(obj)
        df = _build_base_features(p)

        # Decide whether social is enabled for this symbol
        use_social = SOCIAL_ENABLED and (SOCIAL_INCLUDE is None or sym_u in SOCIAL_INCLUDE)

        if use_social:
            df = _merge_social(df, social_hourly)
        else:
            # Ensure the column exists and is zero when disabled
            df["social_score"] = 0.0

        # Final cleanup: fill remaining NaNs with 0 for model safety
        for col in ["r_1h", "r_3h", "r_6h", "vol_6h", "atr_14h", "sma_gap", "high_vol", "social_score"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].astype(float).fillna(0.0)

        out[sym_u] = df[["ts", "price", "r_1h", "r_3h", "r_6h", "vol_6h", "atr_14h", "sma_gap", "high_vol", "social_score"]]

    return out
