# scripts/ml/social_features.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
import os, json
import pandas as pd

def _load_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["created_utc"])
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def _hour_floor(ts: pd.Series) -> pd.Series:
    # expects ISO strings with Z
    return pd.to_datetime(ts.str.replace("Z", "+00:00"), utc=True).dt.floor("H")

def _series_from_logs(df: pd.DataFrame, kind: str) -> pd.Series:
    """
    Turns social logs into an hourly score in [0,1].
    For now: min-max normalize per-hour counts -> score; neutral = 0.5 fallback.
    """
    if df.empty:
        return pd.Series(dtype="float64")

    if kind == "reddit":
        ts = _hour_floor(df["created_utc"].astype(str))
        counts = ts.value_counts().sort_index()
    else:  # twitter
        ts = _hour_floor(df["created_utc"].astype(str))
        counts = ts.value_counts().sort_index()

    if counts.empty:
        return pd.Series(dtype="float64")

    c = counts.astype("float64")
    lo, hi = c.min(), c.max()
    if hi == lo:
        score = pd.Series(0.5, index=c.index)
    else:
        score = (c - lo) / (hi - lo) * 0.8 + 0.1  # keep away from 0/1 extremes

    score.name = f"{kind}_score"
    return score

def compute_social_series(repo_root: Path = Path(".")) -> pd.DataFrame:
    """
    Returns a single DataFrame indexed hourly with columns:
      ['reddit_score','twitter_score','social_score']
    If disabled or no data → empty df (caller defaults to neutral 0.5).
    Gated by MW_SOCIAL_ENABLED (default off).
    """
    if str(os.getenv("MW_SOCIAL_ENABLED", "0")).lower() not in {"1","true","yes"}:
        return pd.DataFrame()

    logs_dir = repo_root / "logs"
    reddit_df = _load_jsonl(logs_dir / "social_reddit.jsonl")
    tw_df     = _load_jsonl(logs_dir / "social_twitter.jsonl")

    rs = _series_from_logs(reddit_df, "reddit")
    ts = _series_from_logs(tw_df, "twitter")

    df = pd.concat([rs, ts], axis=1).sort_index()
    if "reddit_score" not in df: df["reddit_score"] = 0.5
    if "twitter_score" not in df: df["twitter_score"] = 0.5
    df["social_score"] = df[["reddit_score","twitter_score"]].mean(axis=1).fillna(0.5)
    return df