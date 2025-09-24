# scripts/market/ingest_market.py
from __future__ import annotations

import math
import os
import time
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.summary_sections.common import ensure_dir  # type: ignore

try:
    from scripts.market.coingecko_client import CoinGeckoClient  # type: ignore
except Exception:
    CoinGeckoClient = None  # type: ignore


@dataclass
class Cfg:
    base_url: str
    api_key: Optional[str]
    coins: List[str]
    vs: str
    lookback_h: int
    rate_per_min: int
    demo: bool
    artifacts_dir: Path


def _default_base_url(api_key: Optional[str]) -> str:
    # If key present, prefer pro; else public.
    return "https://pro-api.coingecko.com/api/v3" if api_key else "https://api.coingecko.com/api/v3"


def _env_cfg() -> Cfg:
    coins = [c.strip().lower() for c in os.getenv("MW_CG_COINS", "bitcoin,ethereum,solana").split(",") if c.strip()]
    api_key = os.getenv("MW_CG_API_KEY") or None
    # If MW_CG_BASE_URL not set, infer from presence of api key.
    base_url = os.getenv("MW_CG_BASE_URL") or _default_base_url(api_key)
    return Cfg(
        base_url=base_url,
        api_key=api_key,
        coins=coins,
        vs=os.getenv("MW_CG_VS_CURRENCY", "usd").lower(),
        lookback_h=int(os.getenv("MW_CG_LOOKBACK_H", "72")),
        rate_per_min=int(os.getenv("MW_CG_RATE_LIMIT_PER_MIN", "25")),
        demo=str(os.getenv("MW_DEMO", "")).lower() in ("1", "true", "yes"),
        artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", "artifacts")),
    )


def _fmt_usd(p: float) -> str:
    return f"${float(p):,.2f}"


def _epoch_hour_floor(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _hourly_timeline(now_utc: datetime, lookback_h: int) -> List[datetime]:
    end = _epoch_hour_floor(now_utc)
    start = end - timedelta(hours=lookback_h - 1)
    cur = start
    out: List[datetime] = []
    while cur <= end:
        out.append(cur)
        cur += timedelta(hours=1)
    return out


def _resample_hourly_from_pairs(pairs_ms_price: List[List[float]], now_utc: datetime, lookback_h: int) -> List[Tuple[int, float]]:
    if not pairs_ms_price:
        return []
    pts = sorted([(int(ms) // 1000, float(px)) for ms, px in pairs_ms_price], key=lambda x: x[0])
    timeline = _hourly_timeline(now_utc, lookback_h)
    out: List[Tuple[int, float]] = []
    i = 0
    latest_px: Optional[float] = None
    for bucket_dt in timeline:
        bucket_s = int(bucket_dt.timestamp())
        while i < len(pts) and pts[i][0] <= bucket_s:
            latest_px = pts[i][1]
            i += 1
        if latest_px is not None:
            out.append((bucket_s, latest_px))
    return out


def _hourly_returns(series: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    out: List