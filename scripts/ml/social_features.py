# scripts/ml/social_features.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
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


def _to_hour_floor_any(series: pd.Series) -> pd.Series:
    """
    Robust timestamp parser:
      - Accepts ISO8601 ('...Z' or '+00:00') strings
      - Accepts epoch seconds (int/float/str)
    Returns UTC hourly floor timestamps.
    """
    # Try ISO strings
    a = pd.to_datetime(
        series.astype(str).str.replace("Z", "+00:00"),
        utc=True,
        errors="coerce",
    )
    # Try epoch seconds
    b = pd.to_datetime(
        pd.to_numeric(series, errors="coerce"),
        unit="s",
        utc=True,
        errors="coerce",
    )
    ts = a.fillna(b)
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
    Count events per hour from a DataFrame with a 'created_utc' timestamp column.
    Accepts ISO strings or epoch seconds.
    """
    if df.empty or time_col not in df:
        return pd.Series(dtype="int64")
    hours = _to_hour_floor_any(df[time_col])
    vc = hours.value_counts().sort_index()
    vc.index.name = "hour"
    return vc


def _normalized_from_counts(counts: pd.Series) -> pd.Series:
    """
    Prefer a rolling z-score mapped to [0,1]; fall back to min-max if needed.
    """
    if counts is None or counts.empty:
        return pd.Series(dtype="float64")

    # Ensure continuous hourly index
    counts = counts.asfreq("h").fillna(0).astype("float64")

    # Rolling window ~30d of hours; require at least 24 hours to start
    roll = counts.rolling(window=24 * 30, min_periods=24)
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
def _reddit_series(reddit_df: pd.DataFrame) -> pd.Series:
    """
    Build a normalized reddit_score from hourly post counts, then lag by +1 hour (anti-leak).
    """
    if reddit_df.empty:
        return pd.Series(dtype="float64")

    counts = _hourly_counts(reddit_df, "created_utc")
    if counts.empty:
        return pd.Series(dtype="float64")

    score = _normalized_from_counts(counts)
    score.name = "reddit_score"

    # Anti-leak: shift forward by +1 hour so hour T uses info from T-1 and earlier.
    return score.shift(1)


def _twitter_series(tw_df: pd.DataFrame) -> pd.Series:
    """
    Same approach for Twitter if logs are present; otherwise returns empty.
    """
    if tw_df.empty:
        return pd.Series(dtype="float64")

    counts = _hourly_counts(tw_df, "created_utc")
    if counts.empty:
        return pd.Series(dtype="float64")

    score = _normalized_from_counts(counts)
    score.name = "twitter_score"

    # Anti-leak lag
    return score.shift(1)


def compute_social_series(repo_root: Path = Path(".")) -> pd.DataFrame:
    """
    Returns a DataFrame indexed hourly with columns:
      ['reddit_score','twitter_score','social_score']

    - Gated by MW_SOCIAL_ENABLED (default off).
    - If disabled or no data, returns empty (caller should default to neutral 0.5).
    - Applies a conservative +1h lag to avoid information leakage.
    """
    if str(os.getenv("MW_SOCIAL_ENABLED", "0")).lower() not in {"1", "true", "yes"}:
        return pd.DataFrame()

    # Coerce in case caller passed a str (e.g., ".")
    repo_root = Path(repo_root)

    logs_dir = repo_root / "logs"
    reddit_df = _load_jsonl(logs_dir / "social_reddit.jsonl")
    tw_df = _load_jsonl(logs_dir / "social_twitter.jsonl")

    rs = _reddit_series(reddit_df).rename("reddit_score")
    ts = _twitter_series(tw_df).rename("twitter_score")

    df = pd.concat([rs, ts], axis=1).sort_index()

    # Ensure both columns exist
    if "reddit_score" not in df.columns:
        df["reddit_score"] = pd.Series(dtype="float64")
    if "twitter_score" not in df.columns:
        df["twitter_score"] = pd.Series(dtype="float64")

    # Combine → social_score
    df["social_score"] = df[["reddit_score", "twitter_score"]].mean(axis=1)

    # Ensure hourly continuity from min..max and fill neutral
    if not df.empty:
        # Ensure tz-aware index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h", tz="UTC")
        df = df.reindex(idx)

    df = df.fillna(0.5)

    # Only keep intended columns (avoid any stray unnamed column)
    return df[["reddit_score", "twitter_score", "social_score"]]