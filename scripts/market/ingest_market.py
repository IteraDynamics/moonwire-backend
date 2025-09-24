# scripts/market/ingest_market.py
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

# matplotlib: force headless
import matplotlib
matplotlib.use("Agg")  # noqa
import matplotlib.pyplot as plt  # noqa

from .coingecko_client import CoinGeckoClient


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _iso(dt: datetime) -> str:
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


def _demo_series(now: datetime, hours: int) -> List[Dict[str, Any]]:
    """
    Create seed-friendly synthetic hourly price series (monotone-ish with wiggles).
    """
    out = []
    base = 1.0
    for h in range(hours):
        t = now - timedelta(hours=(hours - 1 - h))
        # simple shape: 1 + small sin wave + mild trend
        price = base * (100 + 2.5 * math.sin(h / 4.0) + 0.2 * h)
        out.append({"t": int(t.replace(tzinfo=timezone.utc).timestamp()), "price": float(round(price, 6))})
    return out


def _nearest_return(series: List[Dict[str, Any]], lookback_h: int) -> float:
    """
    % return over ~lookback_h hours using nearest timestamp <= target.
    series: list of {'t': seconds, 'price': float} sorted by t asc.
    """
    if not series:
        return 0.0
    last = series[-1]
    target_ts = last["t"] - lookback_h * 3600
    # find nearest index with t >= target_ts (or last fallback)
    idx = 0
    for i, pt in enumerate(series):
        if pt["t"] >= target_ts:
            idx = i
            break
    start = series[idx]
    p0 = float(start["price"])
    p1 = float(last["price"])
    if p0 <= 0:
        return 0.0
    return (p1 - p0) / p0


def _plot_price(coin: str, series: List[Dict[str, Any]], out_path: Path) -> None:
    if not series:
        return
    xs = [datetime.fromtimestamp(pt["t"], tz=timezone.utc) for pt in series]
    ys = [pt["price"] for pt in series]
    plt.figure(figsize=(7, 3.2), dpi=140)
    plt.plot(xs, ys, lw=1.8)
    plt.title(f"{coin.upper()} price (last {len(series)} pts)")
    plt.xlabel("time (UTC)")
    plt.ylabel("price")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_returns(coin: str, series: List[Dict[str, Any]], out_path: Path) -> None:
    if not series:
        return
    # simple hourly returns vs previous point
    rs = []
    for i in range(1, len(series)):
        p0 = float(series[i - 1]["price"])
        p1 = float(series[i]["price"])
        rs.append(0.0 if p0 == 0 else (p1 - p0) / p0)
    xs = [datetime.fromtimestamp(pt["t"], tz=timezone.utc) for pt in series[1:]]
    plt.figure(figsize=(7, 3.2), dpi=140)
    plt.plot(xs, rs, lw=1.4)
    plt.title(f"{coin.upper()} hourly returns")
    plt.xlabel("time (UTC)")
    plt.ylabel("return")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _build_from_live(coins: List[str], vs: str, lookback_h: int) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    """
    Fetch live data from CoinGecko and normalize to the expected structure.
    Returns (simple_price_data, series_map)
    """
    client = CoinGeckoClient()
    # 1) latest spot
    simple = client.simple_price(coins, vs, True)

    # 2) time series
    series: Dict[str, List[Dict[str, Any]]] = {}
    for c in coins:
        chart = client.market_chart_hours(c, vs, lookback_h)
        prices = chart.get("prices") or []
        # prices is list [[ms, price], ...]; make hourly-ish
        pts: List[Dict[str, Any]] = []
        for ms, p in prices:
            try:
                t = int(ms // 1000)
                pts.append({"t": t, "price": float(p)})
            except Exception:
                # skip malformed points
                continue
        # trim to last lookback_h hours (best effort)
        if pts:
            cutoff = pts[-1]["t"] - lookback_h * 3600
            pts = [pt for pt in pts if pt["t"] >= cutoff]
        series[c] = pts

    return simple, series


def build_market_context(
    coins: List[str],
    vs_currency: str,
    lookback_h: int,
    models_dir: Path,
    artifacts_dir: Path,
) -> Dict[str, Any]:
    """
    Main entrypoint used by summary section. Returns the JSON dict (also writes it/plots).
    - Reads MW_DEMO to decide live vs demo (but always *writes* artifacts).
    - On live failure, falls back to demo and records demo_reason.
    """
    _ensure_dir(models_dir)
    _ensure_dir(artifacts_dir)

    now = datetime.now(timezone.utc)
    demo_env = (os.getenv("MW_DEMO", "").lower() == "true")
    demo = False
    demo_reason = None

    series: Dict[str, List[Dict[str, Any]]] = {}
    live_spot: Dict[str, Any] = {}

    if not demo_env:
        try:
            live_spot, series = _build_from_live(coins, vs_currency, lookback_h)
            # sanity: ensure we got at least some points; otherwise force demo fallback
            if not all(series.get(c) for c in coins):
                raise ValueError("missing series for one or more coins")
        except Exception as e:
            demo = True
            demo_reason = f"live_fetch_failed: {type(e).__name__}"
            # fall through to demo series
    else:
        demo = True
        demo_reason = "demo_requested"

    if demo:
        # synth series pivoted per coin; keep lookback_h points
        for c in coins:
            series[c] = _demo_series(now, max(lookback_h, 12))
        # fabricate a spot snapshot roughly matching last series point
        live_spot = {c: {vs_currency: series[c][-1]["price"], f"{vs_currency}_24h_change": 0.0} for c in coins}

    # compute headline returns
    rets: Dict[str, Dict[str, float]] = {}
    for c in coins:
        s = series.get(c, [])
        rets[c] = {
            "h1": round(_nearest_return(s, 1), 6),
            "h24": round(_nearest_return(s, 24), 6),
            "h72": round(_nearest_return(s, 72), 6),
        }

    # write JSON artifact
    out: Dict[str, Any] = {
        "generated_at": _iso(now),
        "vs": vs_currency,
        "coins": coins,
        "window_hours": lookback_h,
        "series": series,
        "returns": rets,
        "demo": bool(demo),
        "attribution": "CoinGecko",
    }
    if demo and demo_reason:
        out["demo_reason"] = demo_reason

    jpath = models_dir / "market_context.json"
    jpath.write_text(json.dumps(out), encoding="utf-8")

    # plots (always)
    for c in coins:
        _plot_price(c, series.get(c, []), artifacts_dir / f"market_trend_price_{c}.png")
    for c in coins:
        _plot_returns(c, series.get(c, []), artifacts_dir / f"market_trend_returns_{c}.png")

    return out


def run_ingest(logs_dir: Path, models_dir: Path, artifacts_dir: Path) -> Dict[str, Any]:
    """
    Thin wrapper used by tests.
    Env:
      - MW_CG_COINS (default: "bitcoin,ethereum,solana")
      - MW_CG_VS_CURRENCY (default: "usd")
      - MW_CG_LOOKBACK_H (default: "72")
      - MW_DEMO ("true" to force demo)
    """
    _ensure_dir(logs_dir)       # not currently used, but created for parity with tests
    _ensure_dir(models_dir)
    _ensure_dir(artifacts_dir)

    coins = [c.strip() for c in os.getenv("MW_CG_COINS", "bitcoin,ethereum,solana").split(",") if c.strip()]
    vs = os.getenv("MW_CG_VS_CURRENCY", "usd").strip().lower()
    lookback_h = int(os.getenv("MW_CG_LOOKBACK_H", "72"))

    return build_market_context(coins, vs, lookback_h, models_dir, artifacts_dir)