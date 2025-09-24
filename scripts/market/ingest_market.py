# scripts/market/ingest_market.py
from __future__ import annotations

import math
import os
import time
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Matplotlib headless for CI
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Reuse Summary helpers (paths, ensure_dir)
from scripts.summary_sections.common import ensure_dir  # type: ignore

# Optional client; we handle ImportError by demo fallback
try:
    from scripts.market.coingecko_client import CoinGeckoClient  # type: ignore
except Exception:  # pragma: no cover - import may fail in some CI setups
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


def _env_cfg() -> Cfg:
    coins = [c.strip().lower() for c in os.getenv("MW_CG_COINS", "bitcoin,ethereum,solana").split(",") if c.strip()]
    return Cfg(
        base_url=os.getenv("MW_CG_BASE_URL", "https://pro-api.coingecko.com/api/v3"),
        api_key=os.getenv("MW_CG_API_KEY") or None,
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
    """
    Input: [[ts_ms, price], ...] (possibly irregular); Output: hourly [(epoch_s, price)] for last H hours.
    Strategy: last observation carried forward within each hour; if no history before bucket, skip.
    """
    if not pairs_ms_price:
        return []
    # Normalize & sort
    pts = sorted([(int(ms) // 1000, float(px)) for ms, px in pairs_ms_price], key=lambda x: x[0])

    # Map latest point <= bucket
    timeline = _hourly_timeline(now_utc, lookback_h)
    out: List[Tuple[int, float]] = []
    i = 0
    latest_px: Optional[float] = None
    for bucket_dt in timeline:
        bucket_s = int(bucket_dt.timestamp())
        # advance pointer up to bucket
        while i < len(pts) and pts[i][0] <= bucket_s:
            latest_px = pts[i][1]
            i += 1
        if latest_px is not None:
            out.append((bucket_s, latest_px))
        else:
            # no data yet; skip this bucket
            pass
    return out


def _hourly_returns(series: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Simple returns r_t = p_t / p_{t-1} - 1 for aligned hourly series."""
    out: List[Tuple[int, float]] = []
    for idx in range(1, len(series)):
        t, p = series[idx]
        _, prev = series[idx - 1]
        if prev != 0:
            out.append((t, p / prev - 1.0))
    return out


def _window_return(series: List[Tuple[int, float]], hours: int) -> Optional[float]:
    """Return over last N hours from most recent point; None if not enough history."""
    if not series:
        return None
    tail = series[-1]
    tail_t, tail_p = tail
    target_t = tail_t - hours * 3600
    # Find the first point at or before target_t
    prev_p = None
    for t, p in reversed(series):
        if t <= target_t:
            prev_p = p
            break
    if prev_p is None or prev_p == 0:
        return None
    return tail_p / prev_p - 1.0


def _plot_price(coin: str, series: List[Tuple[int, float]], out_path: Path, lookback_h: int):
    ensure_dir(out_path.parent)
    if not series:
        return
    xs = [datetime.fromtimestamp(t, tz=timezone.utc) for t, _ in series]
    ys = [p for _, p in series]
    plt.figure(figsize=(8.5, 3))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    plt.title(f"{coin} price ({lookback_h}h)")
    plt.xlabel("UTC time")
    plt.ylabel("price")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_returns(coin: str, returns: List[Tuple[int, float]], out_path: Path, lookback_h: int):
    ensure_dir(out_path.parent)
    if not returns:
        return
    xs = [datetime.fromtimestamp(t, tz=timezone.utc) for t, _ in returns]
    ys = [r for _, r in returns]
    plt.figure(figsize=(8.5, 3))
    plt.axhline(0.0)
    plt.bar(xs, ys)
    plt.title(f"{coin} hourly returns ({lookback_h}h)")
    plt.xlabel("UTC time")
    plt.ylabel("return")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _seed_demo_series(coins: List[str], now_utc: datetime, lookback_h: int) -> Dict[str, List[Tuple[int, float]]]:
    """
    Deterministic sine + noise per coin. Prices are vaguely realistic but synthetic.
    """
    random.seed(1337)
    base = {
        "bitcoin": 60000.0,
        "ethereum": 3000.0,
        "solana": 150.0,
    }
    amp = {
        "bitcoin": 4000.0,
        "ethereum": 200.0,
        "solana": 12.0,
    }
    timeline = _hourly_timeline(now_utc, lookback_h)
    out: Dict[str, List[Tuple[int, float]]] = {}
    for c in coins:
        b = base.get(c, 100.0)
        a = amp.get(c, 5.0)
        pts: List[Tuple[int, float]] = []
        for i, dt in enumerate(timeline):
            # 2-day sine cycle + gentle drift + tiny noise (deterministic)
            theta = 2.0 * math.pi * (i / 48.0)
            px = b * (1.0 + 0.01 * math.sin(theta)) + a * math.sin(theta * 1.7) + 0.25 * a * math.sin(theta * 0.33)
            px = max(0.1, px)
            pts.append((int(dt.timestamp()), float(px)))
        out[c] = pts
    return out


def _max_sleep_for_rate(rate_per_min: int):
    # leave a safety margin; we also batch endpoints where possible.
    if rate_per_min <= 0:
        return 0.0
    return max(60.0 / float(rate_per_min), 0.0) * 1.10


def _live_fetch_series(cfg: Cfg, coins: List[str], now_utc: datetime) -> Dict[str, List[Tuple[int, float]]]:
    """
    Fetch hourly series per coin using /coins/{id}/market_chart days=ceil(H/24)
    """
    if CoinGeckoClient is None:
        raise ImportError("CoinGeckoClient unavailable")

    client = CoinGeckoClient(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        max_per_min=cfg.rate_per_min,
        timeout_connect=5,
        timeout_read=10,
        max_retries=4,
    )

    days = max(1, math.ceil(cfg.lookback_h / 24))
    series: Dict[str, List[Tuple[int, float]]] = {}
    spacing = _max_sleep_for_rate(cfg.rate_per_min)

    for i, coin in enumerate(coins):
        # Respect pacing across requests
        if i > 0 and spacing > 0:
            time.sleep(spacing)

        # /market_chart returns {"prices": [[ts_ms, price], ...], ...}
        data = client.get_market_chart(coin_id=coin, vs_currency=cfg.vs, days=days)
        pairs = data.get("prices") or []
        ser = _resample_hourly_from_pairs(pairs, now_utc=now_utc, lookback_h=cfg.lookback_h)
        series[coin] = ser

    return series


def _append_spot_log(logs_dir: Path, vs: str, spots: Dict[str, float], source: str, demo: bool):
    ensure_dir(logs_dir)
    path = logs_dir / "market_prices.jsonl"
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    lines = []
    for coin, price in spots.items():
        row = {
            "ts_utc": ts,
            "id": coin,
            "symbol": coin[:3],  # cosmetic
            "vs": vs,
            "price": float(price),
            "source": "coingecko",
            "attribution": "CoinGecko",
            "demo": bool(demo),
        }
        lines.append(json.dumps(row))
    with path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _latest_spots_from_series(series: Dict[str, List[Tuple[int, float]]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for c, ser in series.items():
        if ser:
            out[c] = float(ser[-1][1])
    return out


def _symbol_upper(coin_id: str) -> str:
    # Lightweight mapping; fallback to first 3 letters
    if coin_id == "bitcoin":
        return "BTC"
    if coin_id == "ethereum":
        return "ETH"
    if coin_id == "solana":
        return "SOL"
    return coin_id[:3].upper()


def build_market_context(ctx) -> Tuple[Dict[str, object], List[str]]:
    """
    Public entry point used by summary_sections.market_context.

    Side-effects:
      - writes models/market_context.json
      - writes artifacts/market_trend_price_<coin>.png
      - writes artifacts/market_trend_returns_<coin>.png
      - appends logs/market_prices.jsonl
    Returns:
      (context_json_dict, markdown_lines)
    """
    cfg = _env_cfg()
    now_utc = datetime.now(timezone.utc)

    # Decide live vs demo
    demo_reason: Optional[str] = None
    use_demo = cfg.demo
    series: Dict[str, List[Tuple[int, float]]] = {}

    if not use_demo:
        try:
            series = _live_fetch_series(cfg, cfg.coins, now_utc=now_utc)
            # Sanity: if any coin has no points, fall back
            if not any(series.get(c) for c in cfg.coins):
                raise RuntimeError("no series returned")
        except Exception as e:
            use_demo = True
            demo_reason = f"live_fetch_failed: {e.__class__.__name__}"

    if use_demo:
        series = _seed_demo_series(cfg.coins, now_utc=now_utc, lookback_h=cfg.lookback_h)

    # Compute returns + aggregates
    returns_by_coin: Dict[str, List[Tuple[int, float]]] = {}
    agg_returns: Dict[str, Dict[str, Optional[float]]] = {}
    for c in cfg.coins:
        ser = series.get(c, [])
        rets = _hourly_returns(ser)
        returns_by_coin[c] = rets
        agg_returns[c] = {
            "h1": (rets[-1][1] if len(rets) >= 1 else None),
            "h24": _window_return(ser, 24),
            "h72": _window_return(ser, 72),
        }

    # Write JSON artifact
    models_dir: Path = ensure_dir(Path(getattr(ctx, "models_dir", "models")))
    payload: Dict[str, object] = {
        "generated_at": now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "vs": cfg.vs,
        "coins": cfg.coins,
        "window_hours": cfg.lookback_h,
        "series": {c: [{"t": int(t), "price": float(p)} for (t, p) in series.get(c, [])] for c in cfg.coins},
        "returns": {
            c: {k: (None if v is None else round(float(v), 6)) for k, v in agg_returns.get(c, {}).items()}
            for c in cfg.coins
        },
        "demo": bool(use_demo),
        "attribution": "CoinGecko",
    }
    if demo_reason:
        payload["demo_reason"] = demo_reason

    (models_dir / "market_context.json").write_text(json.dumps(payload), encoding="utf-8")

    # Append logs (spot)
    spots = _latest_spots_from_series(series)
    logs_dir: Path = ensure_dir(Path(getattr(ctx, "logs_dir", "logs")))
    _append_spot_log(logs_dir, vs=cfg.vs, spots=spots, source="coingecko", demo=use_demo)

    # Plots
    art_dir = ensure_dir(cfg.artifacts_dir)
    for c in cfg.coins:
        ser = series.get(c, [])
        rets = returns_by_coin.get(c, [])
        _plot_price(c, ser, art_dir / f"market_trend_price_{c}.png", cfg.lookback_h)
        _plot_returns(c, rets, art_dir / f"market_trend_returns_{c}.png", cfg.lookback_h)

    # Markdown block
    header = f"📈 Market Context (CoinGecko, {cfg.lookback_h}h)" + (" (demo)" if use_demo else "")
    lines: List[str] = [header]
    # crude volatility hint: mark vol ↑ if |h24| > 2%
    for c in cfg.coins:
        sym = _symbol_upper(c)
        px = spots.get(c)
        r = agg_returns.get(c, {})
        h1 = r.get("h1")
        h24 = r.get("h24")
        h72 = r.get("h72")
        vol_flag = " [vol ↑]" if (h24 is not None and abs(h24) >= 0.02) else ""
        price_str = _fmt_usd(px) if px is not None else "—"
        def pct(v): return ("—" if v is None else f"{v*100:+.1f}%")
        lines.append(f"• {sym} → {price_str} | h1 {pct(h1)} | h24 {pct(h24)} | h72 {pct(h72)}{vol_flag}")
    lines.append("— Data via CoinGecko API; subject to plan rate limits.")

    # If demo fallback occurred with a reason, expose it subtly
    if demo_reason:
        lines[-1] += f" [demo_fallback: {demo_reason}]"

    return payload, lines