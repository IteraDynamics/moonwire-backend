# src/ml/feature_builder.py
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# Use the canonical social series builder from scripts (already anti-leak lagged).
# PYTHONPATH in CI includes repo root, so this import is valid.
try:
    from scripts.ml.social_features import compute_social_series
except Exception:
    compute_social_series = None  # handled gracefully below


# --- Social gating helpers (add near imports) ---
def _parse_csv_env(name: str) -> set[str]:
    raw = os.getenv(name, "") or ""
    return {s.strip().upper() for s in raw.split(",") if s.strip()}

# Global gate + allow/deny lists:
_SOC_INCLUDE = _parse_csv_env("MW_SOCIAL_INCLUDE")
_GLOBAL_SOC_ON = str(os.getenv("MW_SOCIAL_ENABLED", "0")).lower() in {"1","true","yes"}


def _should_use_social(symbol: str) -> bool:
    sym = (symbol or "").upper()
    if _SOC_INCLUDE:
        # INCLUDE list is authoritative
        return sym in _SOC_INCLUDE
    return _GLOBAL_SOC_ON
    


# ----------------------------
# Helpers (timestamps & parsing)
# ----------------------------
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
# Social merge (canonical path)
# ----------------------------
def _load_social_hourly(repo_root: Path = Path(".")) -> pd.DataFrame:
    """
    Load the canonical hourly social series:
      index (UTC hourly), columns ['reddit_score','twitter_score','social_score'].
    Returns empty df if disabled or unavailable.
    """
    if not _env_bool("MW_SOCIAL_ENABLED", False):
        return pd.DataFrame()
    if compute_social_series is None:
        return pd.DataFrame()
    try:
        df = compute_social_series(repo_root)
        # Ensure hourly tz-aware index for robust joins
        if df is not None and not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            df = df.asfreq("h").fillna(0.5)
        return df
    except Exception:
        return pd.DataFrame()


def _attach_social_score(df: pd.DataFrame, social_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach 'social_score' to the per-row frame by hourly key. If disabled or missing,
    default to neutral 0.5. No leakage: the canonical series is already lagged +1h.
    """
    df = df.copy()
    hours = _as_hour_series(df)
    df["social_score"] = 0.5

    if hours is None or social_df is None or social_df.empty:
        return df

    s = social_df.get("social_score")
    if s is None or s.empty:
        return df

    # Build the exact hours we need, reindex social, and map row-wise.
    need_idx = pd.to_datetime(pd.Series(hours), utc=True, errors="coerce")
    uniq_hours = sorted(need_idx.dropna().unique())
    if len(uniq_hours) == 0:
        return df

    s_re = s.astype(float).reindex(uniq_hours).fillna(0.5)
    mapper = dict(zip(uniq_hours, s_re.values.tolist()))
    df["social_score"] = [float(mapper.get(h, 0.5)) for h in hours]

    return df


# ----------------------------
# Price-derived features
# ----------------------------
def _features_for_symbol(df: pd.DataFrame, sym: str, social_df: pd.DataFrame) -> pd.DataFrame:
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
    df = _attach_social_score(df, social_df)

    # clean
    df = df.dropna().reset_index(drop=True)
    return df


def _write_feature_manifest(frames: Dict[str, pd.DataFrame], out_path: Path = Path("models/ml_model_manifest.json")) -> None:
    """
    Non-destructively update the model manifest with the union of feature columns
    actually produced by the builder (for CI verification).
    """
    feats: List[str] = sorted(set().union(*[set(df.columns.tolist()) for df in frames.values() if not df.empty]))
    try:
        if out_path.exists():
            j = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            j = {}
        j["features"] = feats
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(j, indent=2), encoding="utf-8")
    except Exception:
        # best-effort only; don't break training
        pass


def build_features(df_prices: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Build per-symbol feature frames.

    Price-only by default.
    If MW_SOCIAL_ENABLED=1 and social logs exist, attach aligned hourly social_score
    from the canonical lagged social series.

    Returns: dict[symbol] -> DataFrame
    """
    # Load shared social time-series once
    social_df = _load_social_hourly(Path("."))


# Per-symbol social gating
for _sym, _df in out.items():  # replace 'out' with your actual dict
    if "social_score" in _df.columns and not _should_use_social(_sym):
        _df["social_score"] = 0.0

return out
   

    out: Dict[str, pd.DataFrame] = {}
    for sym, df in df_prices.items():
        out[sym] = _features_for_symbol(df, sym, social_df)

    # Write features list for CI verification (non-blocking)
    _write_feature_manifest(out)

    return out
