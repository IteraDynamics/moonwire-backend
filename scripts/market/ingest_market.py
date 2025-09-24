# scripts/market/ingest_market.py
from __future__ import annotations

import os
import json
import math
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta, timezone

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # CI-safe
import matplotlib.pyplot as plt  # noqa: E402

from .coingecko_client import CoinGeckoClient


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)


def _to_unix(dt: datetime) -> int:
    return int(dt.timestamp())


def _fmt_money(p: float, symbol: str = "$") -> str:
    try:
        return f"{symbol}{float(p):,.2f}"
    except Exception:
        return f"{symbol}0.00"


def _write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _append_jsonl(path: Path, line: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line) + "\n")


def _resample_hourly(prices_ms: List[Tuple[int, float]], lookback_h: int) -> List[Tuple[datetime, float]]:
    """
    Input: list of (ts_ms, price) sorted ascending or not.
    Output: hourly (right-labeled) points covering last lookback_h hours.
    We keep the last price within each hour bucket.
    """
    if not prices_ms:
        return []
    prices_ms = sorted(prices_ms, key=lambda x: x[0])
    now = _now_utc()
    start = now - timedelta(hours=lookback_h)
    # Build buckets on the hour
    buckets: Dict[datetime, float] = {}
    for ts_ms, price in prices_ms:
        try:
            ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
        except Exception:
            continue
        if ts < start or ts > now:
            continue
        hour = _floor_to_hour(ts)
        buckets[hour] = float(price)  # last wins

    # Ensure contiguous hours
    out: List[Tuple[datetime, float]] = []
    h = _floor_to_hour(start)
    end_hour = _floor_to_hour(now)
    last_price: float | None = None
    while h <= end_hour:
        if h in buckets:
            last_price = buckets[h]
        if last_price is not None:
            out.append((h, last_price))
        h = h + timedelta(hours=1)
    return out


def _hourly_returns(series: List[Tuple[datetime, float]]) -> List[Tuple[datetime, float]]:
    out: List[Tuple[datetime, float]] = []
    prev = None
    for t, p in series:
        if prev is None:
            out.append((t, 0.0))
        else:
            try:
                r = (p / prev) - 1.0 if prev else 0.0
            except Exception:
                r = 0.0
            out.append((t, r))
        prev = p
    return out


def _window_return(series: List[Tuple[datetime, float]], hours: int) -> float:
    if not series:
        return 0.0
    end_t, end_p = series[-1]
    start_cut = end_t - timedelta(hours=hours)
    # find first point at/after start_cut
    start_p = None
    for t, p in series:
        if t >= start_cut:
            start_p = p
            break
    if start_p is None:
        # fallback to earliest
        start_p = series[0][1]
    try:
        return (end_p / start_p) - 1.0 if start_p else 0.0
    except Exception:
        return 0.0


def _plot_price(path: Path, series: List[Tuple[datetime, float]], title: str):
    if not series:
        return
    xs = [t for t, _ in series]
    ys = [p for _, p in series]
    plt.figure()
    plt.plot(xs, ys, marker="o", linestyle="-", linewidth=1)
    plt.title(title)
    plt.xlabel("time (UTC)")
    plt.ylabel("price")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _plot_returns(path: Path, rets: List[Tuple[datetime, float]], title: str):
    if not rets:
        return
    xs = [t for t, _ in rets]
    ys = [r * 100.0 for _, r in rets]  # percent
    plt.figure()
    plt.bar(xs, ys)
    # zero line
    plt.axhline(0.0)
    plt.title(title)
    plt.xlabel("time (UTC)")
    plt.ylabel("hourly return (%)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _demo_series(coin_id: str, lookback_h: int) -> List[Tuple[datetime, float]]:
    # Deterministic-ish sine with small noise
    now = _now_utc()
    start = now - timedelta(hours=lookback_h)
    out: List[Tuple[datetime, float]] = []
    base = {"bitcoin": 60000.0, "ethereum": 3000.0, "solana": 150.0}.get(coin_id, 100.0)
    amp = base * 0.05
    step = timedelta(hours=1)
    h = _floor_to_hour(start)
    i = 0
    while h <= _floor_to_hour(now):
        val = base + amp * math.sin(i / 6.0) + amp * 0.1 * math.sin(i / 2.0)
        out.append((h, float(val)))
        h += step
        i += 1
    return out


def build_market_context(models_dir: Path, artifacts_dir: Path, logs_dir: Path) -> Dict[str, Any]:
    """
    Live ingestion if possible; otherwise demo fallback.
    Returns the JSON object written to models/market_context.json
    """
    vs = os.getenv("MW_CG_VS_CURRENCY", "usd").strip().lower() or "usd"
    coins = [c.strip().lower() for c in (os.getenv("MW_CG_COINS", "bitcoin,ethereum,solana").split(","))]
    coins = [c for c in coins if c]
    lookback_h = int(os.getenv("MW_CG_LOOKBACK_H", "72") or "72")
    demo_mode = os.getenv("MW_DEMO", "").strip().lower() in ("1", "true", "yes")

    series_map: Dict[str, List[Tuple[datetime, float]]] = {}
    spot_map: Dict[str, float] = {}
    used_demo = False
    demo_reason = None

    if not demo_mode:
        client = CoinGeckoClient()
        try:
            # 1) Spot (single call)
            sp = client.simple_price(coins, vs, include_24h_change=True)
            for cid in coins:
                val = sp.get(cid, {}).get(vs)
                if isinstance(val, (int, float)):
                    spot_map[cid] = float(val)

            # 2) History per-coin
            days = max(1, math.ceil(lookback_h / 24))
            for cid in coins:
                try:
                    prices = client.market_chart_days(cid, vs, days)
                    series_map[cid] = _resample_hourly(prices, lookback_h)
                except Exception as e:
                    # per-coin fallback: at least have something
                    series_map[cid] = _demo_series(cid, lookback_h)
                    if demo_reason is None:
                        demo_reason = f"partial_history_failed[{cid}]: {type(e).__name__}"
            client.close()
        except Exception as e:
            used_demo = True
            demo_reason = f"live_fetch_failed: {type(e).__name__}"
            # Demo for all coins
            for cid in coins:
                series_map[cid] = _demo_series(cid, lookback_h)
                # synthetic spot = last point
                if series_map[cid]:
                    spot_map[cid] = series_map[cid][-1][1]
    else:
        used_demo = True
        demo_reason = "MW_DEMO=true"
        for cid in coins:
            series_map[cid] = _demo_series(cid, lookback_h)
            if series_map[cid]:
                spot_map[cid] = series_map[cid][-1][1]

    # Compute returns and JSON artifact
    json_series: Dict[str, List[Dict[str, Any]]] = {}
    returns_summary: Dict[str, Dict[str, float]] = {}
    for cid in coins:
        ser = series_map.get(cid, [])
        rets = _hourly_returns(ser)
        json_series[cid] = [{"t": _to_unix(t), "price": float(p)} for t, p in ser]
        returns_summary[cid] = {
            "h1": float(rets[-1][1]) if len(rets) >= 1 else 0.0,
            "h24": float(_window_return(ser, 24)),
            "h72": float(_window_return(ser, 72)),
        }

    obj = {
        "generated_at": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "vs": vs,
        "coins": coins,
        "window_hours": lookback_h,
        "series": json_series,
        "returns": returns_summary,
        "demo": bool(used_demo),
        "attribution": "CoinGecko",
    }
    if used_demo and demo_reason:
        obj["demo_reason"] = demo_reason

    # Write JSON
    _ensure_dir(models_dir)
    _write_json(models_dir / "market_context.json", obj)

    # Append spot logs
    _ensure_dir(logs_dir)
    ts = obj["generated_at"]
    for cid in coins:
        price = spot_map.get(cid)
        if isinstance(price, (int, float)):
            _append_jsonl(
                logs_dir / "market_prices.jsonl",
                {
                    "ts_utc": ts,
                    "id": cid,
                    "symbol": {"bitcoin": "btc", "ethereum": "eth", "solana": "sol"}.get(cid, cid[:3]),
                    "vs": vs,
                    "price": float(price),
                    "source": "coingecko",
                    "attribution": "CoinGecko",
                    "demo": bool(used_demo),
                },
            )

    # Plots
    _ensure_dir(artifacts_dir)
    for cid in coins:
        ser = series_map.get(cid, [])
        if not ser:
            continue
        _plot_price(artifacts_dir / f"market_trend_price_{cid}.png", ser, f"{cid} price ({obj['window_hours']}h)")
        rets = _hourly_returns(ser)
        _plot_returns(artifacts_dir / f"market_trend_returns_{cid}.png", rets, f"{cid} hourly returns ({obj['window_hours']}h)")

    return obj


def run_ingest(models_dir: str | Path, artifacts_dir: str | Path, logs_dir: str | Path) -> Dict[str, Any]:
    return build_market_context(Path(models_dir), Path(artifacts_dir), Path(logs_dir))
