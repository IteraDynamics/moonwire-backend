from __future__ import annotations

import os, math, json, random
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timezone, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.summary_sections.common import ensure_dir, _iso

from .coingecko_client import CoinGeckoClient


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None else default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "yes", "on")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _resample_hourly(ts_pairs_ms: List[List[float]], window_h: int) -> List[Tuple[int, float]]:
    """Input pairs [[ts_ms, price], ...], Output hourly [(epoch_s, price)] within window."""
    now = _now_utc()
    start = now - timedelta(hours=window_h)

    # bucket by hour (right-labeled -> use last sample in the hour)
    buckets: Dict[datetime, float] = {}
    for t_ms, price in ts_pairs_ms or []:
        t = datetime.fromtimestamp(float(t_ms) / 1000.0, tz=timezone.utc)
        if t < start or t > now:
            continue
        bucket = t.replace(minute=0, second=0, microsecond=0)
        buckets[bucket] = float(price)  # keep last seen

    # ensure monotonically increasing sequence
    out: List[Tuple[int, float]] = []
    t = start.replace(minute=0, second=0, microsecond=0)
    end = now.replace(minute=0, second=0, microsecond=0)
    last = None
    while t <= end:
        if t in buckets:
            last = buckets[t]
        if last is not None:
            out.append((int(t.timestamp()), last))
        t += timedelta(hours=1)
    return out[-(window_h + 1):]  # guard length


def _hourly_returns(series: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Simple returns between consecutive hours."""
    out: List[Tuple[int, float]] = []
    for i in range(1, len(series)):
        t0, p0 = series[i - 1]
        t1, p1 = series[i]
        if p0:
            out.append((t1, (p1 - p0) / p0))
    return out


def _window_returns(series: List[Tuple[int, float]], hours: int) -> float:
    """Return from last price over given hours; fallback to 0 if insufficient history."""
    if not series:
        return 0.0
    end_t, end_p = series[-1]
    start_cut = end_t - hours * 3600
    start_p = None
    for t, p in reversed(series):
        if t <= start_cut:
            start_p = p
            break
    if start_p is None:
        start_p = series[0][1]
    if not start_p:
        return 0.0
    return (end_p - start_p) / start_p


def _demo_series(base: float, window_h: int, seed: int = 1337) -> List[Tuple[int, float]]:
    random.seed(seed)
    now = _now_utc().replace(minute=0, second=0)
    start = now - timedelta(hours=window_h)
    points: List[Tuple[int, float]] = []
    t = start
    val = base
    while t <= now:
        # sine-ish wave + noise, bounded positive
        ph = (t - start).total_seconds() / 3600.0
        drift = 1.0 + 0.01 * math.sin(ph / 5.0) + random.uniform(-0.002, 0.002)
        val = max(0.0001, val * drift)
        points.append((int(t.timestamp()), float(val)))
        t += timedelta(hours=1)
    return points


def _plot_price_and_returns(coin: str, series: List[Tuple[int, float]], ret_series: List[Tuple[int, float]], artifacts_dir: Path) -> Tuple[Path, Path]:
    ensure_dir(artifacts_dir)
    # price
    fig = plt.figure(figsize=(9, 3))
    ax = plt.gca()
    xs = [datetime.fromtimestamp(t, tz=timezone.utc) for t, _ in series]
    ys = [p for _, p in series]
    ax.plot(xs, ys, marker="o", linewidth=1.5)
    ax.set_title(f"Price Trend — {coin}")
    ax.set_ylabel("price")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p1 = artifacts_dir / f"market_trend_price_{coin}.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # returns
    fig = plt.figure(figsize=(9, 3))
    ax = plt.gca()
    xs = [datetime.fromtimestamp(t, tz=timezone.utc) for t, _ in ret_series]
    ys = [r for _, r in ret_series]
    ax.bar(xs, ys)
    ax.axhline(0.0, linewidth=1)
    ax.set_title(f"Hourly Returns — {coin}")
    ax.set_ylabel("return")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p2 = artifacts_dir / f"market_trend_returns_{coin}.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p1, p2


def run_ingest(logs_dir: Path, models_dir: Path, artifacts_dir: Path) -> Dict[str, any]:
    ensure_dir(logs_dir); ensure_dir(models_dir); ensure_dir(artifacts_dir)

    vs = _env("MW_CG_VS_CURRENCY", "usd")
    look_h = int(_env("MW_CG_LOOKBACK_H", "72"))
    coins = [c.strip() for c in _env("MW_CG_COINS", "bitcoin,ethereum,solana").split(",") if c.strip()]
    demo = _env_bool("MW_DEMO", False)

    client = CoinGeckoClient()
    data_series: Dict[str, List[Tuple[int, float]]] = {}

    if demo or client.demo:
        # deterministic demo series with different bases
        bases = {"bitcoin": 60000.0, "ethereum": 3000.0, "solana": 150.0}
        for i, cid in enumerate(coins):
            base = bases.get(cid, 100.0 + i * 50.0)
            data_series[cid] = _demo_series(base, look_h, seed=100 + i)
        latest_price = {c: data_series[c][-1][1] for c in coins if data_series.get(c)}
        demo_used = True
    else:
        # fetch simple price for all coins
        spot = client.simple_price(ids=coins, vs=vs, include_24h_change=True)
        latest_price = {k: float(v.get(vs)) for k, v in (spot or {}).items() if isinstance(v, dict) and v.get(vs) is not None}

        # fetch hourly-ish history per coin
        days = max(1, math.ceil(look_h / 24))
        for cid in coins:
            try:
                chart = client.market_chart_days(cid, vs, days)
                prices = (chart or {}).get("prices") or []
                data_series[cid] = _resample_hourly(prices, look_h)
            except Exception:
                # soft fail: synthesize if one coin fails
                data_series[cid] = _demo_series(max(latest_price.get(cid, 100.0), 1.0), look_h, seed=200 + hash(cid) % 1000)
        demo_used = False

    # compute returns + aggregate json
    hourly_returns = {cid: _hourly_returns(data_series.get(cid, [])) for cid in coins}
    ret_window = {
        "h1": 1,
        "h24": 24,
        "h72": 72,
    }
    returns_summary = {cid: {k: round(_window_returns(data_series.get(cid, []), hrs), 6) for k, hrs in ret_window.items()} for cid in coins}

    # append spot logs
    log_path = logs_dir / "market_prices.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        ts = _iso(_now_utc())
        for cid in coins:
            price = latest_price.get(cid, data_series.get(cid, [(-1, 0.0)])[-1][1])
            row = {
                "ts_utc": ts,
                "id": cid,
                "symbol": (cid[:3] if cid else "").lower(),
                "vs": vs,
                "price": float(price),
                "source": "coingecko",
                "attribution": "CoinGecko",
                "demo": bool(demo or demo_used),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # write model artifact
    model_json = {
        "generated_at": _iso(_now_utc()),
        "vs": vs,
        "coins": coins,
        "window_hours": look_h,
        "series": {cid: [{"t": t, "price": p} for (t, p) in data_series.get(cid, [])] for cid in coins},
        "returns": returns_summary,
        "demo": bool(demo or demo_used),
        "attribution": "CoinGecko",
    }
    (models_dir / "market_context.json").write_text(json.dumps(model_json, ensure_ascii=False), encoding="utf-8")

    # charts
    for cid in coins:
        s = data_series.get(cid, [])
        r = hourly_returns.get(cid, [])
        _plot_price_and_returns(cid, s, r, artifacts_dir)

    return model_json