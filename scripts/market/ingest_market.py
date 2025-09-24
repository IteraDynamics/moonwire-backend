# scripts/market/ingest_market.py
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from .coingecko_client import CoinGeckoClient


# --------- helpers ---------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _iso(dt: Optional[datetime] = None) -> str:
    dt = dt or _now_utc()
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _write_json(path: Path, data: Dict) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"), sort_keys=False)

def _append_jsonl(path: Path, row: Dict) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x*100:.1f}%"

def _fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


# --------- demo series generator ---------

def _demo_series(coin: str, lookback_h: int, base_price: float) -> List[Tuple[int, float]]:
    """
    Deterministic sine-ish series per coin with light noise.
    Hourly points over the last lookback_h hours (inclusive of current hour).
    """
    seed = abs(hash(coin)) % (2**32)
    rng = random.Random(seed)
    now = _now_utc().replace(minute=0, second=0, microsecond=0)
    points: List[Tuple[int, float]] = []
    amp = 0.04  # 4% swing
    for k in range(lookback_h + 1):
        t = now - timedelta(hours=lookback_h - k)
        # smooth daily-ish wave + tiny coin-specific offset
        phase = (k / 24.0) * 2 * math.pi + (seed % 360) * math.pi / 180.0
        wiggle = 1.0 + amp * math.sin(phase)
        noise = 1.0 + rng.uniform(-0.005, 0.005)
        price = base_price * wiggle * noise
        points.append((_epoch(t), float(price)))
    return points


# --------- resampling + returns ---------

def _to_hourly(points: List[Tuple[int, float]], lookback_h: int) -> List[Tuple[int, float]]:
    """
    Right-labeled hourly buckets over the last lookback_h hours ending now (rounded down to hour).
    If multiple raw points fall into the same hour, use the last observed.
    If an hour is missing, forward-fill from the previous hour if available.
    """
    if not points:
        return []

    # map hour -> price (take last in that hour)
    by_hour: Dict[int, float] = {}
    for ts, p in points:
        hour = (ts // 3600) * 3600
        by_hour[hour] = float(p)

    now_hr = (_epoch(_now_utc()) // 3600) * 3600
    start_hr = now_hr - lookback_h * 3600
    out: List[Tuple[int, float]] = []
    last = None
    h = start_hr
    while h <= now_hr:
        if h in by_hour:
            last = by_hour[h]
        if last is not None:
            out.append((h, last))
        else:
            # no data yet; skip until first available
            pass
        h += 3600
    return out

def _window_return(series: List[Tuple[int, float]], hours_back: int) -> Optional[float]:
    if not series:
        return None
    end_ts, end_p = series[-1]
    target_ts = end_ts - hours_back * 3600
    # find the first point >= target_ts (or closest earlier if missing)
    prev = None
    for ts, p in series:
        if ts == target_ts:
            prev = p
            break
        if ts > target_ts:
            prev = p  # already ffilled by construction
            break
        prev = p
    if prev is None:
        return None
    if prev <= 0:
        return None
    return (end_p - prev) / prev

def _hourly_returns(series: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """
    Return per-hour simple returns aligned to each hour (r_t = p_t/p_{t-1}-1), first hour -> 0.0
    """
    if not series:
        return []
    out: List[Tuple[int, float]] = []
    prev = None
    for ts, p in series:
        if prev is None or prev <= 0:
            out.append((ts, 0.0))
        else:
            out.append((ts, (p - prev) / prev))
        prev = p
    return out


# --------- core builders ---------

@dataclass
class MarketBuildResult:
    context: Dict
    summary_md: str
    demo: bool
    demo_reason: Optional[str] = None


def build_market_context(
    models_dir: Path,
    artifacts_dir: Path,
    coins: List[str],
    vs: str = "usd",
    lookback_h: int = 72,
    demo: bool = False,
    demo_reason: Optional[str] = None,
) -> MarketBuildResult:
    """
    Build models/market_context.json + plots. Returns context + md.
    """
    _ensure_dir(models_dir)
    _ensure_dir(artifacts_dir)

    series: Dict[str, List[Tuple[int, float]]] = {}
    returns_h: Dict[str, Dict[str, Optional[float]]] = {}
    spot_prices: Dict[str, float] = {}

    if not demo:
        client = CoinGeckoClient()
        # spot (optional 24h change not required for tests)
        spot = client.simple_price(coins, vs_currency=vs, include_24h_change=False)
        for c in coins:
            v = spot.get(c, {})
            price = float(v.get(vs, 0.0))
            if price <= 0:
                raise ValueError(f"missing spot for {c}")
            spot_prices[c] = price

        # history
        for c in coins:
            raw = client.history_last_hours(c, vs_currency=vs, lookback_hours=lookback_h)
            ser = _to_hourly(raw, lookback_h)
            if not ser:
                raise ValueError(f"empty history for {c}")
            series[c] = ser

    else:
        # demo: deterministic base per coin
        bases = {"bitcoin": 60363.14, "ethereum": 3025.84, "solana": 150.18}
        for c in coins:
            base = bases.get(c, 100.0 + abs(hash(c)) % 500)
            ser = _demo_series(c, lookback_h, float(base))
            ser = _to_hourly(ser, lookback_h)
            if not ser:
                raise ValueError("demo series unexpectedly empty")
            series[c] = ser
            spot_prices[c] = ser[-1][1]

    # compute returns & write plots
    for c in coins:
        ser = series[c]
        r1 = _window_return(ser, 1)
        r24 = _window_return(ser, 24) if lookback_h >= 24 else None
        r72 = _window_return(ser, 72) if lookback_h >= 72 else None
        returns_h[c] = {"h1": r1, "h24": r24, "h72": r72}

        # plots
        # price
        ts = [t for t, _ in ser]
        ps = [p for _, p in ser]
        plt.figure()
        plt.plot([datetime.fromtimestamp(t, tz=timezone.utc) for t in ts], ps, marker="o", markersize=2)
        plt.title(f"{c} price ({lookback_h}h)")
        plt.xlabel("time (UTC)")
        plt.ylabel(f"price ({vs})")
        price_png = artifacts_dir / f"market_trend_price_{c}.png"
        plt.tight_layout()
        plt.savefig(price_png)
        plt.close()

        # returns
        rets = _hourly_returns(ser)
        rt = [t for t, _ in rets]
        rv = [v for _, v in rets]
        plt.figure()
        plt.axhline(0.0)
        plt.bar([datetime.fromtimestamp(t, tz=timezone.utc) for t in rt], rv)
        plt.title(f"{c} hourly returns ({lookback_h}h)")
        plt.xlabel("time (UTC)")
        plt.ylabel("return")
        ret_png = artifacts_dir / f"market_trend_returns_{c}.png"
        plt.tight_layout()
        plt.savefig(ret_png)
        plt.close()

    # write JSON artifact
    ctx = {
        "generated_at": _iso(),
        "vs": vs,
        "coins": coins,
        "window_hours": lookback_h,
        "series": {
            c: [{"t": t, "price": p} for t, p in series[c]] for c in coins
        },
        "returns": {
            c: {k: (float(v) if v is not None else None) for k, v in returns_h[c].items()}
            for c in coins
        },
        "demo": bool(demo),
        "attribution": "CoinGecko",
    }
    if demo and demo_reason:
        ctx["demo_reason"] = demo_reason

    _write_json(models_dir / "market_context.json", ctx)

    # build markdown summary block
    lines = []
    tag = "(demo)" if demo else ""
    lines.append(f"📈 Market Context (CoinGecko, {lookback_h}h) {tag}".strip())
    code_to_symbol = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL"}
    for c in coins:
        sym = code_to_symbol.get(c, c[:3].upper())
        price = spot_prices[c]
        r = returns_h[c]
        h1 = _fmt_pct(r["h1"])
        h24 = _fmt_pct(r["h24"])
        h72 = _fmt_pct(r["h72"])
        # cheap volatility hint
        vol_hint = ""
        if r["h24"] is not None and abs(r["h24"]) >= 0.02:
            vol_hint = " [vol ↑]"
        lines.append(f"• {sym} → {_fmt_usd(price)} | h1 {h1} | h24 {h24} | h72 {h72}{vol_hint}")
    lines.append("— Data via CoinGecko API; subject to plan rate limits.")
    if demo and demo_reason:
        lines.append(f"[demo_fallback: {demo_reason}]")

    md = "\n".join(lines)
    return MarketBuildResult(context=ctx, summary_md=md, demo=demo, demo_reason=demo_reason)


# --------- public entrypoint used by tests / CI ---------

def run_ingest(logs_dir: Path, models_dir: Path, artifacts_dir: Path) -> MarketBuildResult:
    """
    Orchestrates live-or-demo build, appends spot lines to logs, and returns build result.
    Never raises on live failure: will fall back to demo.
    """
    # config
    coins = [c.strip() for c in os.getenv("MW_CG_COINS", "bitcoin,ethereum,solana").split(",") if c.strip()]
    vs = os.getenv("MW_CG_VS_CURRENCY", "usd").strip().lower() or "usd"
    lookback_h = int(os.getenv("MW_CG_LOOKBACK_H", "72"))
    demo_env = (os.getenv("MW_DEMO", "").lower() == "true")

    # try live unless forced demo
    result: Optional[MarketBuildResult] = None
    if not demo_env:
        try:
            result = build_market_context(models_dir, artifacts_dir, coins, vs=vs, lookback_h=lookback_h, demo=False)
        except Exception as e:
            # fallback to demo
            result = build_market_context(
                models_dir, artifacts_dir, coins, vs=vs, lookback_h=lookback_h, demo=True,
                demo_reason=f"live_fetch_failed: {e.__class__.__name__}"
            )
    else:
        result = build_market_context(models_dir, artifacts_dir, coins, vs=vs, lookback_h=lookback_h, demo=True)

    # append logs (one line per coin)
    ts = _iso()
    for c in result.context["coins"]:
        last_price = result.context["series"][c][-1]["price"]
        row = {
            "ts_utc": ts,
            "id": c,
            "symbol": {"bitcoin": "btc", "ethereum": "eth", "solana": "sol"}.get(c, c[:3].lower()),
            "vs": vs,
            "price": float(last_price),
            "source": "coingecko",
            "attribution": "CoinGecko",
            "demo": bool(result.demo),
        }
        _append_jsonl(Path(logs_dir) / "market_prices.jsonl", row)

    return result


__all__ = ["run_ingest", "build_market_context"]