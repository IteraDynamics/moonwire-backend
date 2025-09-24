# scripts/market/ingest_market.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

import math

# Matplotlib for charts (headless fine under CI with MPLBACKEND=Agg)
import matplotlib.pyplot as plt

# Optional: live client (tests run in demo path; live path is used in CI summary)
try:
    from . import coingecko_client as cg
except Exception:  # pragma: no cover - tests don't need live client import to succeed
    cg = None


# ----------------------------
# Small utilities
# ----------------------------

def _iso(dt: datetime | None = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b


# ----------------------------
# Demo data builder (deterministic-ish)
# ----------------------------

def _demo_series(start_price: float, points: int, step_minutes: int = 60) -> List[Tuple[int, float]]:
    """
    Build a simple wavy series with at least `points` entries.
    Returns [(unix_sec, price), ...] in **ascending** time order.
    """
    now = datetime.now(timezone.utc)
    series = []
    for i in range(points):
        t = now - timedelta(minutes=step_minutes * (points - 1 - i))
        # gentle wave + drift
        price = start_price * (1.0 + 0.005 * math.sin(i / 2.0)) * (1.0 + 0.0005 * i)
        series.append((int(t.timestamp()), round(price, 2)))
    return series


def _build_demo_context(coins: List[str], lookback_h: int, vs: str) -> Dict[str, Any]:
    # Ensure >= 12 hourly points for tests
    points = max(lookback_h, 12)
    seeds = {
        "bitcoin": 60363.14,
        "ethereum": 3025.84,
        "solana": 150.18,
    }
    series = {}
    for c in coins:
        base = seeds.get(c, 100.0)
        series[c] = [{"t": t, "price": p} for t, p in _demo_series(base, points, step_minutes=60)]

    # compute 1h / 24h / 72h returns from tail if enough history, else 0
    returns = {}
    for c, arr in series.items():
        last = arr[-1]["price"]
        def lag(h):
            idx = max(0, len(arr) - 1 - h)
            return arr[idx]["price"]
        returns[c] = {
            "h1": _pct(last, lag(1)) if len(arr) > 1 else 0.0,
            "h24": _pct(last, lag(24)) if len(arr) > 24 else 0.0,
            "h72": _pct(last, lag(72)) if len(arr) > 72 else 0.0,
        }

    return {
        "generated_at": _iso(),
        "vs": vs,
        "coins": coins,
        "window_hours": lookback_h,
        "series": {c: [{"t": pt["t"], "price": pt["price"]} for pt in series[c]] for c in coins},
        "returns": returns,
        "demo": True,
        "demo_reason": os.getenv("MW_DEMO_REASON") or "forced_demo_in_tests",
        "attribution": "CoinGecko",
    }


# ----------------------------
# Live data builder
# ----------------------------

def _build_live_context(coins: List[str], lookback_h: int, vs: str) -> Dict[str, Any]:
    if cg is None:
        raise RuntimeError("coingecko_client unavailable")
    client = cg.CoinGeckoClient()
    # prices now (+ 24h change if exposed)
    sp = client.simple_price(coins, vs, include_24h_change=True) or {}
    # hourly history via 1-day market_chart (coingecko returns ~hourly buckets)
    series: Dict[str, List[Dict[str, Any]]] = {}
    for c in coins:
        m = client.market_chart_days(c, vs, days=1)
        pts = []
        if isinstance(m, dict) and "prices" in m:
            for ts_ms, price in m["prices"]:
                # coingecko gives ms epoch
                pts.append({"t": int(ts_ms / 1000), "price": float(price)})
            # only last `lookback_h` hours
            pts = pts[-lookback_h:]
        series[c] = pts

    # compute returns (use available history)
    returns = {}
    for c, arr in series.items():
        if not arr:
            returns[c] = {"h1": 0.0, "h24": 0.0, "h72": 0.0}
            continue
        last = arr[-1]["price"]
        def lag(h):
            idx = max(0, len(arr) - 1 - h)
            return arr[idx]["price"]
        returns[c] = {
            "h1": _pct(last, lag(1)) if len(arr) > 1 else 0.0,
            "h24": _pct(last, lag(24)) if len(arr) > 24 else 0.0,
            "h72": _pct(last, lag(72)) if len(arr) > 72 else 0.0,
        }

    return {
        "generated_at": _iso(),
        "vs": vs,
        "coins": coins,
        "window_hours": lookback_h,
        "series": series,
        "returns": returns,
        "demo": False,
        "attribution": "CoinGecko",
    }


# ----------------------------
# Public builder (used by summary section import)
# ----------------------------

def build_market_context() -> Dict[str, Any]:
    """
    Build a {series, returns, ...} dict either from live CoinGecko or demo.
    Controlled by environment:
      - MW_DEMO=true => force demo
      - MW_CG_COINS (default: 'bitcoin,ethereum,solana')
      - MW_CG_VS_CURRENCY (default: 'usd')
      - MW_CG_LOOKBACK_H (default: '72')
    """
    coins = [c.strip() for c in (os.getenv("MW_CG_COINS") or "bitcoin,ethereum,solana").split(",") if c.strip()]
    vs = (os.getenv("MW_CG_VS_CURRENCY") or "usd").lower().strip()
    lookback_h = int(os.getenv("MW_CG_LOOKBACK_H") or "72")

    force_demo = (os.getenv("MW_DEMO") or "").lower() == "true"
    if force_demo:
        return _build_demo_context(coins, lookback_h, vs)

    # Try live; on any failure, fall back to demo but record reason
    try:
        ctx = _build_live_context(coins, lookback_h, vs)
        # sanity: ensure we have at least minimal points, else consider as failure
        if not all(ctx["series"].get(c) for c in coins):
            raise RuntimeError("empty series from live fetch")
        return ctx
    except Exception as e:  # pragma: no cover (exercised in CI, not unit tests)
        os.environ["MW_DEMO_REASON"] = f"live_fetch_failed: {type(e).__name__}"
        return _build_demo_context(coins, lookback_h, vs)


# ----------------------------
# Plot helpers
# ----------------------------

def _plot_price(coin: str, rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    xs = [datetime.fromtimestamp(r["t"], tz=timezone.utc) for r in rows]
    ys = [r["price"] for r in rows]
    plt.figure()
    plt.plot(xs, ys)
    plt.title(f"{coin.upper()} price")
    plt.xlabel("time (UTC)")
    plt.ylabel("price")
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path)
    plt.close()


def _plot_returns(coin: str, rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    # simple hourly returns
    rs = []
    for i in range(1, len(rows)):
        a = rows[i]["price"]
        b = rows[i - 1]["price"]
        rs.append(_pct(a, b))
    xs = list(range(1, len(rows)))
    plt.figure()
    plt.plot(xs, rs)
    plt.title(f"{coin.upper()} hourly returns")
    plt.xlabel("hour index (relative)")
    plt.ylabel("return (fraction)")
    plt.tight_layout()
    _ensure_dir(path.parent)
    plt.savefig(path)
    plt.close()


# ----------------------------
# Ingest entry point used by tests
# ----------------------------

def run_ingest(log_dir: Path, models_dir: Path, artifacts_dir: Path) -> Dict[str, Any]:
    """
    Build context (demo or live), save JSON/PNGs, and **append one line** to logs/market_prices.jsonl.
    """
    ctx = build_market_context()

    # 1) Save JSON artifact
    _write_json(models_dir / "market_context.json", ctx)

    # 2) Save charts (per coin)
    coins: List[str] = ctx.get("coins", [])
    series: Dict[str, List[Dict[str, Any]]] = ctx.get("series", {})
    for c in coins:
        rows = series.get(c, [])
        _plot_price(c, rows, artifacts_dir / f"market_trend_price_{c}.png")
        _plot_returns(c, rows, artifacts_dir / f"market_trend_returns_{c}.png")

    # 3) Append a log line (even in DEMO) so tests can assert existence
    prices_now = {}
    for c in coins:
        arr = series.get(c, [])
        prices_now[c] = arr[-1]["price"] if arr else None

    log_row = {
        "generated_at": ctx.get("generated_at") or _iso(),
        "vs": ctx.get("vs", "usd"),
        "coins": coins,
        "prices": prices_now,
        "demo": bool(ctx.get("demo", False)),
        "source": "coingecko_demo" if ctx.get("demo") else "coingecko_live",
    }
    _append_jsonl(log_dir / "market_prices.jsonl", log_row)

    return ctx


# CLI usage: optional manual run
if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--models", default="models")
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()
    run_ingest(Path(args.logs), Path(args.models), Path(args.artifacts))
    print("Ingest complete.")