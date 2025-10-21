# scripts/ml/social_features.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def _to_hour_floor_iso(series: pd.Series) -> pd.Series:
    """
    Convert ISO8601 '...Z' strings to UTC hourly floor timestamps.
    """
    ts = pd.to_datetime(series.astype(str).str.replace("Z", "+00:00"), utc=True, errors="coerce")
    return ts.dt.floor("H")


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

    # Optional robustness: use 30-day rolling zscore to stabilize scaling if a long backfill is present.
    # Fallback to min-max if rolling stats are degenerate.
    counts = counts.asfreq("H").fillna(0)
    roll = counts.rolling(window=24 * 30, min_periods=24)  # ~30d
    mean = roll.mean()
    std = roll.std(ddof=0)
    with pd.option_context("mode.use_inf_as_na", True):
        z = (counts - mean) / std
    if z.dropna().empty:
        s01 = _minmax_01(counts)
    else:
        # map z to [0,1] via CDF-ish clipping
        zc = z.clip(-3, 3)  # -3..3
        s01 = (zc - zc.min()) / (zc.max() - zc.min()).replace(0, 1)

    score = _squash_away_from_extremes(s01)  # map to ~[0.1,0.9]
    score.name = "reddit_score"

    # Anti-leak: shift forward by +1 hour so hour T uses info from T-1 and earlier.
    score = score.shift(1)

    return score


def _twitter_series(tw_df: pd.DataFrame) -> pd.Series:
    """
    Same approach for Twitter if logs are present; otherwise returns empty.
    """
    if tw_df.empty:
        return pd.Series(dtype="float64")

    counts = _hourly_counts(tw_df, "created_utc")
    if counts.empty:
        return pd.Series(dtype="float64")

    counts = counts.asfreq("H").fillna(0)
    roll = counts.rolling(window=24 * 30, min_periods=24)
    mean = roll.mean()
    std = roll.std(ddof=0)
    with pd.option_context("mode.use_inf_as_na", True):
        z = (counts - mean) / std
    if z.dropna().empty:
        s01 = _minmax_01(counts)
    else:
        zc = z.clip(-3, 3)
        s01 = (zc - zc.min()) / (zc.max() - zc.min()).replace(0, 1)

    score = _squash_away_from_extremes(s01)
    score.name = "twitter_score"

    # Anti-leak lag
    score = score.shift(1)

    return score


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

    logs_dir = repo_root / "logs"
    reddit_df = _load_jsonl(logs_dir / "social_reddit.jsonl")
    tw_df = _load_jsonl(logs_dir / "social_twitter.jsonl")

    rs = _reddit_series(reddit_df)
    ts = _twitter_series(tw_df)

    df = pd.concat([rs, ts], axis=1).sort_index()
    if "reddit_score" not in df:
        df["reddit_score"] = pd.Series(dtype="float64")
    if "twitter_score" not in df:
        df["twitter_score"] = pd.Series(dtype="float64")

    # Combine (simple mean) and fill missing with neutral 0.5
    df["social_score"] = df[["reddit_score", "twitter_score"]].mean(axis=1)
    df = df.fillna(0.5)

    # Ensure hourly frequency continuity (helps feature_builder alignment)
    if not df.empty:
        idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq="H", tz="UTC")
        df = df.reindex(idx).fillna(0.5)

    return df