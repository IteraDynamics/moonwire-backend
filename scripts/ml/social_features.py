# scripts/ml/social_features.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ----------------------------
# Small env helpers
# ----------------------------
def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name)
        return int(raw) if raw not in (None, "") else default
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.getenv(name)
        return float(raw) if raw not in (None, "") else default
    except Exception:
        return default


def _env_on(name: str, default: bool = False) -> bool:
    if (v := os.getenv(name)) is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


# ----------------------------
# I/O helpers
# ----------------------------
def _load_jsonl(path: Path) -> pd.DataFrame:
    """
    Load a JSONL file into a DataFrame. If missing/empty, return empty with created_utc column.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["created_utc"])
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                # skip malformed lines
                continue
    if not rows:
        return pd.DataFrame(columns=["created_utc"])
    return pd.DataFrame(rows)


# ----------------------------
# Basic transforms
# ----------------------------
def _to_hour_floor_iso(series: pd.Series) -> pd.Series:
    """
    Convert ISO8601 '...Z' strings to UTC hourly floor timestamps.
    """
    ts = pd.to_datetime(series.astype(str).str.replace("Z", "+00:00"), utc=True, errors="coerce")
    # Pandas deprecates "H" in favor of "h"
    return ts.dt.floor("h")


def _minmax_01(s: pd.Series) -> pd.Series:
    """
    Map to [0,1]. If constant or empty -> 0.5.
    """
    if s is None or s.empty:
        return pd.Series(dtype="float64")
    s = s.astype("float64")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi):
        return pd.Series(dtype="float64")
    if hi == lo:
        return pd.Series(0.5, index=s.index, dtype="float64")
    return (s - lo) / (hi - lo)


def _squash_away_from_extremes(x: pd.Series, low: float = 0.1, high: float = 0.9) -> pd.Series:
    """
    Take a [0,1] score and squash into [low, high] to avoid hard 0/1.
    """
    if x is None or x.empty:
        return pd.Series(dtype="float64")
    x = x.clip(0.0, 1.0)
    return x * (high - low) + low


def _hourly_counts(df: pd.DataFrame, time_col: str = "created_utc") -> pd.Series:
    """
    Count events per hour from a DataFrame with an ISO 'created_utc' column.
    """
    if df.empty or time_col not in df:
        return pd.Series(dtype="int64")
    hours = _to_hour_floor_iso(df[time_col])
    vc = hours.value_counts().sort_index()
    vc.index.name = "hour"
    return vc


def _normalized_from_counts(counts: pd.Series, norm_win_h: int = 24 * 30) -> pd.Series:
    """
    Prefer a rolling z-score mapped to [0,1]; fall back to min-max if needed.
    norm_win_h is the rolling window size in HOURS (default 30d = 720h).
    """
    if counts is None or counts.empty:
        return pd.Series(dtype="float64")

    # Ensure continuous hourly index
    counts = counts.asfreq("h").fillna(0).astype("float64")

    # Rolling window; require at least 24 hours to start
    norm_win_h = max(24, int(norm_win_h))
    roll = counts.rolling(window=norm_win_h, min_periods=24)
    mean = roll.mean()
    std = roll.std(ddof=0)

    # z = (x - mean) / std; handle div-by-zero and infs explicitly
    z = (counts - mean) / std
    z = z.replace([np.inf, -np.inf], np.nan)

    if z.dropna().empty:
        s01 = _minmax_01(counts)
    else:
        # clip to reasonable band, then scale to [0,1]
        zc = z.clip(-3, 3)
        zmin, zmax = zc.min(skipna=True), zc.max(skipna=True)
        if pd.isna(zmin) or pd.isna(zmax) or zmax == zmin:
            s01 = _minmax_01(counts)
        else:
            s01 = (zc - zmin) / (zmax - zmin)

    return _squash_away_from_extremes(s01)  # ~[0.1, 0.9]


# ----------------------------
# Core: build social series
# ----------------------------
def _reddit_series(reddit_df: pd.DataFrame, norm_win_h: int) -> pd.Series:
    """
    Build a normalized reddit_score from hourly post counts, then lagged later in the pipeline.
    """
    if reddit_df.empty:
        return pd.Series(dtype="float64")

    counts = _hourly_counts(reddit_df, "created_utc")
    if counts.empty:
        return pd.Series(dtype="float64")

    score = _normalized_from_counts(counts, norm_win_h=norm_win_h)
    score.name = "reddit_score"

    # NOTE: we no longer shift here; lag is applied in compute_social_series
    # based on MW_SOC_LAG_H so it can be tuned per run.
    return score


def _twitter_series(tw_df: pd.DataFrame, norm_win_h: int) -> pd.Series:
    """
    Same approach for Twitter if logs are present; otherwise returns empty.
    """
    if tw_df.empty:
        return pd.Series(dtype="float64")

    counts = _hourly_counts(tw_df, "created_utc")
    if counts.empty:
        return pd.Series(dtype="float64")

    score = _normalized_from_counts(counts, norm_win_h=norm_win_h)
    score.name = "twitter_score"

    return score


def compute_social_series(repo_root: Path = Path(".")) -> pd.DataFrame:
    """
    Returns a DataFrame indexed hourly with columns:
      ['reddit_score','twitter_score','social_score']

    - Gated by MW_SOCIAL_ENABLED (default off).
    - If disabled or no data, returns empty (caller should default to neutral 0.5).
    - Applies lag/smoothing/scale based on env knobs (defaults match previous behavior).
        * MW_SOC_LAG_H       (default +1)   : shift series by +k hours (anti-leak by default)
        * MW_SOC_NORM_WIN_H  (default 720)  : rolling normalization window in hours
        * MW_SOC_SMOOTH      (default 0)    : EMA span (hours); 0 disables
        * MW_SOC_SCALE       (default 1.0)  : multiply final social_score
    """
    # Gate
    if not _env_on("MW_SOCIAL_ENABLED", False):
        return pd.DataFrame()

    # Coerce in case caller passes a str (e.g., ".")
    repo_root = Path(repo_root)

    # New knobs (defaults preserve old behavior)
    soc_lag_h    = _env_int("MW_SOC_LAG_H", 1)           # +1h anti-leak default
    soc_norm_win = _env_int("MW_SOC_NORM_WIN_H", 24 * 30)  # 720h default
    soc_smooth   = _env_int("MW_SOC_SMOOTH", 0)          # 0 = no smoothing
    soc_scale    = _env_float("MW_SOC_SCALE", 1.0)       # 1.0 = no scaling

    logs_dir = repo_root / "logs"
    reddit_df = _load_jsonl(logs_dir / "social_reddit.jsonl")
    tw_df = _load_jsonl(logs_dir / "social_twitter.jsonl")

    rs = _reddit_series(reddit_df, norm_win_h=soc_norm_win)
    ts = _twitter_series(tw_df,   norm_win_h=soc_norm_win)

    # Combine and sort by time
    df = pd.concat([rs, ts], axis=1).sort_index()

    # Ensure required columns exist (typed)
    if "reddit_score" not in df:
        df["reddit_score"] = pd.Series(dtype="float64")
    if "twitter_score" not in df:
        df["twitter_score"] = pd.Series(dtype="float64")

    # Combine (simple mean) and fill missing with neutral 0.5
    df["social_score"] = df[["reddit_score", "twitter_score"]].mean(axis=1)
    df = df.fillna(0.5)

    # Ensure a continuous, tz-aware hourly index (helps feature_builder joins)
    if not df.empty:
        # Make sure index is tz-aware UTC
        if getattr(df.index, "tz", None) is None:
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        # Reindex to every hour across the observed span
        idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h", tz="UTC")
        df = df.reindex(idx).fillna(0.5)

    # Optional smoothing (EMA span in hours)
    if soc_smooth and soc_smooth > 0 and not df.empty:
        for col in ["reddit_score", "twitter_score", "social_score"]:
            if col in df:
                df[col] = df[col].ewm(span=soc_smooth, adjust=False).mean()

    # Apply lag (positive values = push information later => anti-leak)
    if not df.empty and soc_lag_h != 0:
        df = df.shift(soc_lag_h)

    # Apply scale
    if not df.empty and "social_score" in df:
        df["social_score"] = df["social_score"] * soc_scale

    return df