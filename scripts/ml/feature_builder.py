# scripts/ml/feature_builder.py
from __future__ import annotations
import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


# ----------------------------
# Helpers (timestamps & parsing)
# ----------------------------
_ISO = "%Y-%m-%dT%H:%M:%SZ"
_RE_BTC = re.compile(r"\b(bitcoin|btc)\b", re.I)
_RE_ETH = re.compile(r"\b(ethereum|eth)\b", re.I)
_RE_SOL = re.compile(r"\b(solana|sol)\b", re.I)

def _to_utc(dt_str: str) -> datetime:
    """Parse tolerant ISO string ('Z' or '+00:00') → aware UTC datetime."""
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(timezone.utc)

def _to_utc_ts(x) -> pd.Timestamp:
    """
    Convert a variety of inputs (datetime, pd.Timestamp, str) into a UTC-aware
    pandas.Timestamp. If naive → localize UTC; if tz-aware → convert to UTC.
    """
    t = pd.Timestamp(x)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t

def _as_hour_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Return a UTC hourly bucket series aligned to df rows.
    Prefers 'ts' column; falls back to datetime-like index.
    """
    if "ts" in df.columns:
        ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        ts = pd.Series(idx, index=df.index)
    if ts.isna().all():
        return None
    # use lowercase 'h' to avoid FutureWarning
    return ts.dt.floor("h")

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


# ----------------------------
# Social score (optional/gated)
# ----------------------------
def _load_jsonl_safe(path: Path) -> List[dict]:
    try:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return [json.loads(ln) for ln in lines if ln.strip()]
    except Exception:
        return []

def _symbol_match_from_row(sym: str, row: dict) -> bool:
    """
    Cheap symbol mapping based on subreddit/keywords/text (no hard deps).
    - BTC: subreddit 'Bitcoin' OR title/text hits 'bitcoin|btc'
    - ETH: subreddit 'ethtrader' OR hits 'ethereum|eth'
    - SOL: subreddit 'Solana' OR hits 'solana|sol'
    """
    sub = (row.get("subreddit") or "").lower()
    title = (row.get("title") or row.get("text") or "")
    if sym == "BTC":
        if sub == "bitcoin" or _RE_BTC.search(title):
            return True
    elif sym == "ETH":
        if sub == "ethtrader" or _RE_ETH.search(title):
            return True
    elif sym == "SOL":
        if sub == "solana" or _RE_SOL.search(title):
            return True
    return False

def _collect_hour_counts_for_symbol(sym: str, look_hours: List[pd.Timestamp]) -> Dict[pd.Timestamp, int]:
    """
    Build per-hour post/tweet counts within the look window for a symbol.
    Uses:
      - logs/social_reddit.jsonl
      - logs/social_twitter.jsonl
    Falls back to empty counts if files missing.
    """
    logs_dir = Path("logs")
    r_rows = _load_jsonl_safe(logs_dir / "social_reddit.jsonl")
    t_rows = _load_jsonl_safe(logs_dir / "social_twitter.jsonl")

    # Window bounds (inclusive) from feature hours
    if not look_hours:
        return {}

    # tz-safe + lowercase hour
    start_h = _to_utc_ts(min(look_hours)).floor("h")
    end_h   = _to_utc_ts(max(look_hours)).floor("h") + pd.Timedelta(hours=1)

    buckets: Dict[pd.Timestamp, int] = {}

    def bump(dt: datetime):
        # convert to UTC-aware Timestamp and bucket to hour (lowercase 'h')
        h = _to_utc_ts(dt).floor("h")
        if h < start_h or h >= end_h:
            return
        buckets[h] = buckets.get(h, 0) + 1

    # Reddit rows: expect created_utc + optional mode/source fields
    for r in r_rows:
        try:
            if not _symbol_match_from_row(sym, r):
                continue
            ciso = r.get("created_utc")
            if not ciso:
                continue
            bump(_to_utc(ciso))
        except Exception:
            continue

    # Twitter rows: created_utc present in lite ingest
    for r in t_rows:
        try:
            if not _symbol_match_from_row(sym, r):
                continue
            ciso = r.get("created_utc")
            if not ciso:
                continue
            bump(_to_utc(ciso))
        except Exception:
            continue

    return buckets

def _normalize_counts_to_score(hrs: List[pd.Timestamp], counts: Dict[pd.Timestamp, int]) -> pd.Series:
    """
    Map hourly integer counts → [0,1] score with a neutral prior (0.5) and safety for flat series.
    """
    xs = [int(counts.get(h, 0)) for h in hrs]
    if len(xs) == 0:
        return pd.Series([], dtype=float)
    mn, mx = min(xs), max(xs)
    if mx == mn:
        # all same -> return neutral 0.5
        return pd.Series([0.5] * len(xs), index=hrs, dtype=float)
    # min/max normalize with a soft prior (pull towards 0.5 a bit)
    raw = [(x - mn) / (mx - mn) for x in xs]
    score = [0.75 * r + 0.25 * 0.5 for r in raw]  # shrink towards 0.5
    return pd.Series(score, index=hrs, dtype=float)

def _attach_social_score(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    """
    If MW_SOCIAL_ENABLED=1, attach real social_score aligned by hour (left-join).
    Otherwise keep neutral 0.5.
    No leakage: we only use posts within each hour bucket (<= that hour).
    """
    enabled = _env_bool("MW_SOCIAL_ENABLED", False)
    df = df.copy()
    df["social_score"] = 0.5
    if not enabled:
        return df

    # Need hourly buckets from df
    hours = _as_hour_series(df)
    if hours is None:
        return df  # can't align -> stay neutral

    # Build score series for the set of hours present
    unique_hours = sorted(pd.Series(hours).dropna().unique())
    hour_counts = _collect_hour_counts_for_symbol(sym, unique_hours)
    hour_scores = _normalize_counts_to_score(unique_hours, hour_counts)

    # Map row-wise
    df["social_score"] = [float(hour_scores.get(h, 0.5)) for h in hours]
    return df


# ----------------------------
# Price-derived features
# ----------------------------
def _features_for_symbol(df: pd.DataFrame, sym: str) -> pd.DataFrame:
    df = df.copy()

    # basic returns
    df["ret_1h"] = df["close"].pct_change(1)
    df["r_1h"] = df["ret_1h"]
    df["r_3h"] = df["close"].pct_change(3)
    df["r_6h"] = df["close"].pct_change(6)

    # volatility
    df["vol_6h"] = df["ret_1h"].rolling(6).std()

    # ATR approx over 14h (true range on OHLC)
    tr = (df["high"] - df["low"]).abs()
    tr_prev_close = (df["high"] - df["close"].shift(1)).abs().combine(
        (df["low"] - df["close"].shift(1)).abs(), max
    )
    df["atr_14h"] = pd.concat([tr, tr_prev_close], axis=1).max(axis=1).rolling(14).mean()

    # momentum
    df["sma_6h"] = df["close"].rolling(6).mean()
    df["sma_24h"] = df["close"].rolling(24).mean()
    df["sma_gap"] = df["sma_6h"] / df["sma_24h"] - 1.0

    # regime flag
    thresh = df["vol_6h"].quantile(0.75)
    df["high_vol"] = (df["vol_6h"] > thresh).astype(int)

    # social (neutral by default; overwrite if gate enabled & logs exist)
    df = _attach_social_score(df, sym)

    # clean
    df = df.dropna().reset_index(drop=True)
    return df


def build_features(df_prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Price-only by default.
    If MW_SOCIAL_ENABLED=1 and social logs exist, attach aligned hourly social_score.
    """
    out: Dict[str, pd.DataFrame] = {}
    for sym, df in df_prices.items():
        out[sym] = _features_for_symbol(df, sym)
    return out