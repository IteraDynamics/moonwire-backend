# scripts/market/ingest_market.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .coingecko_client import CoinGeckoClient


@dataclass
class IngestConfig:
    coins: List[str]
    vs: str
    lookback_h: int
    window_h: int


def _iso(dt: datetime | None = None) -> str:
    return (dt or datetime.now(timezone.utc)).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a / b) - 1.0


def _compute_returns(series: List[Tuple[int, float]]) -> Dict[str, float]:
    """
    series: list of (unix_ts_sec, price)
    Returns h1/h24/h72 using last point vs points ~1/24/72h earlier (best-effort).
    """
    if not series:
        return {"h1": 0.0, "h24": 0.0, "h72": 0.0}

    by_ts = sorted(series, key=lambda x: x[0])
    t_last, p_last = by_ts[-1]
    targets = {
        "h1": t_last - 3600,
        "h24": t_last - 86400,
        "h72": t_last - 72 * 3600,
    }
    out: Dict[str, float] = {}
    for k, tgt in targets.items():
        # find closest at/after tgt
        prior = None
        for t, p in by_ts:
            if t >= tgt:
                prior = (t, p)
                break
        if prior is None:
            prior = by_ts[0]
        out[k] = _pct(p_last, prior[1])
    return out


def build_market_context(client: CoinGeckoClient, cfg: IngestConfig) -> Dict[str, Any]:
    """
    Fetch live data from CoinGecko and produce a normalized market_context dict.
    """
    coins = cfg.coins
    vs = cfg.vs
    window_h = cfg.window_h

    # simple/price (current snapshot)
    _ = client.simple_price(coins, vs, include_24h_change=True)

    # market_chart for each coin (use 'days' that comfortably covers window_h)
    days = max(1, (window_h + 23) // 24)
    series: Dict[str, List[Dict[str, Any]]] = {}
    returns: Dict[str, Dict[str, float]] = {}

    for coin in coins:
        chart = client.market_chart_days(coin, vs, days)
        # chart["prices"] is [[ms, price], ...]
        pts = []
        for ms, price in chart.get("prices", []):
            # clamp/validate
            try:
                t = int(ms // 1000)
                p = float(price)
            except Exception:
                continue
            pts.append({"t": t, "price": p})
        # restrict to last window_h hours
        if pts:
            cutoff = pts[-1]["t"] - window_h * 3600
            pts = [x for x in pts if x["t"] >= cutoff]
        series[coin] = pts

        # returns
        ret_series = [(x["t"], x["price"]) for x in pts]
        returns[coin] = _compute_returns(ret_series)

    # latest price per coin for headline
    latest_prices: Dict[str, float] = {}
    for c, pts in series.items():
        latest_prices[c] = float(pts[-1]["price"]) if pts else 0.0

    ctx = {
        "generated_at": _iso(),
        "vs": vs,
        "coins": coins,
        "window_hours": window_h,
        "series": series,
        "returns": returns,
        "demo": False,
        "attribution": "CoinGecko",
        "latest": latest_prices,
    }
    return ctx


def build_market_context_demo(cfg: IngestConfig) -> Dict[str, Any]:
    """
    Deterministic synthetic series for CI/demo. No network calls.
    """
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    step = timedelta(hours=1)
    n = cfg.window_h + 1
    series: Dict[str, List[Dict[str, Any]]] = {}
    for coin in cfg.coins:
        pts = []
        base = {
            "bitcoin": 60000.0,
            "ethereum": 3000.0,
            "solana": 150.0,
        }.get(coin, 100.0)
        for i in range(n):
            t = int((now - step * (n - 1 - i)).timestamp())
            # gentle wave so returns aren’t all zero
            price = base * (1.0 + 0.0005 * i) * (1.0 + 0.005 * ((i % 6) - 3))
            pts.append({"t": t, "price": round(price, 6)})
        series[coin] = pts

    returns = {c: _compute_returns([(x["t"], x["price"]) for x in pts]) for c, pts in series.items()}
    latest = {c: (pts[-1]["price"] if pts else 0.0) for c, pts in series.items()}

    return {
        "generated_at": _iso(),
        "vs": cfg.vs,
        "coins": cfg.coins,
        "window_hours": cfg.window_h,
        "series": series,
        "returns": returns,
        "demo": True,
        "attribution": "CoinGecko",
        "latest": latest,
    }


def run_ingest(log_dir: Path, models_dir: Path, artifacts_dir: Path) -> Dict[str, Any]:
    """
    Entry-point used by CI/tests. Writes models/market_context.json.
    Returns the context dict.
    """
    coins = [c.strip() for c in os.getenv("MW_CG_COINS", "bitcoin,ethereum,solana").split(",") if c.strip()]
    vs = os.getenv("MW_CG_VS_CURRENCY", "usd")
    lookback_h = int(os.getenv("MW_CG_LOOKBACK_H", "72"))
    window_h = lookback_h  # align naming with summary

    cfg = IngestConfig(coins=coins, vs=vs, lookback_h=lookback_h, window_h=window_h)

    use_demo = os.getenv("MW_DEMO", "").lower() == "true"

    if use_demo:
        ctx = build_market_context_demo(cfg)
    else:
        try:
            client = CoinGeckoClient()
            ctx = build_market_context(client, cfg)
            ctx["demo"] = False
            ctx.pop("demo_reason", None)
        except Exception as e:
            # Fall back with full reason preserved for CI summary
            reason = f"live_fetch_failed: {e.__class__.__name__}: {e}"
            ctx = build_market_context_demo(cfg)
            ctx["demo"] = True
            ctx["demo_reason"] = reason

    # Persist JSON for other steps
    out_json = models_dir / "market_context.json"
    _write_json(out_json, ctx)

    # (Optional) You could render charts into artifacts_dir here if desired.
    # Tests only assert that market_context.json exists.

    return ctx


if __name__ == "__main__":
    # Local manual run
    here = Path.cwd()
    logs = here / "logs"
    models = here / "models"
    arts = here / "artifacts"
    ctx = run_ingest(logs, models, arts)
    print(f"wrote {models/'market_context.json'}; demo={ctx.get('demo')} reason={ctx.get('demo_reason')}")