# scripts/summary_sections/market_context.py
from __future__ import annotations

import os, math, random
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import SummaryContext, _iso

# --- robust import: prefer real ingest, else fallback to local builder ---
_BUILDER = None
try:
    # Preferred: our ingest module exports this symbol
    from scripts.market.ingest_market import build_market_context as _BUILDER  # type: ignore
except Exception:
    _BUILDER = None


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def _fmt_money(p) -> str:
    try:
        return f"${float(p):,.2f}"
    except Exception:
        return "$-"


def _fmt_pct(x) -> str:
    try:
        return f"{float(x)*100:+.1f}%"
    except Exception:
        return "n/a"


def _artifacts_dir(ctx: SummaryContext) -> Path:
    ad = os.getenv("ARTIFACTS_DIR")
    if ad:
        p = Path(ad)
    else:
        p = Path(ctx.models_dir).parent / "artifacts"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _plot_price(series: List[Dict], title: str, out: Path) -> None:
    if not series:
        return
    xs = [datetime.fromtimestamp(pt["t"], tz=timezone.utc) for pt in series]
    ys = [pt["price"] for pt in series]
    fig = plt.figure(figsize=(8, 2.6))
    ax = plt.gca()
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_ylabel("price")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_returns(series: List[float], times: List[datetime], title: str, out: Path) -> None:
    if not series or not times:
        return
    fig = plt.figure(figsize=(8, 2.6))
    ax = plt.gca()
    ax.axhline(0.0, linewidth=1)
    ax.bar(times, series, width=0.030)  # ~hour width
    ax.set_title(title)
    ax.set_ylabel("return")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --- Local demo-only builder (import fallback) ---
def _build_market_context_local(force_demo: bool = True) -> Tuple[Dict, str]:
    """
    Deterministic synthetic series builder matching models/market_context.json schema.
    Used only if the ingest module function isn't importable.
    """
    coins_env = os.getenv("MW_CG_COINS", "bitcoin,ethereum,solana")
    coins = [c.strip().lower() for c in coins_env.split(",") if c.strip()]
    vs = os.getenv("MW_CG_VS_CURRENCY", "usd").lower()
    window_h = int(os.getenv("MW_CG_LOOKBACK_H", "72"))

    # 1h buckets
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=window_h - 1)
    hours = [start + timedelta(hours=i) for i in range(window_h)]

    # deterministic seed
    random.seed(1337)

    series: Dict[str, List[Dict]] = {}
    returns: Dict[str, Dict[str, float]] = {}

    base_map = {"bitcoin": 60000.0, "ethereum": 3000.0, "solana": 150.0}
    amp_map = {"bitcoin": 3500.0, "ethereum": 180.0, "solana": 12.0}

    for coin in coins:
        base = base_map.get(coin, 100.0)
        amp = amp_map.get(coin, 5.0)
        pts: List[Dict] = []
        for i, t in enumerate(hours):
            # slow sine + tiny noise
            theta = (i / 12.0) * math.pi
            price = base + amp * math.sin(theta) + random.random() * (amp * 0.02)
            pts.append({"t": int(t.timestamp()), "price": float(price)})
        series[coin] = pts

        # returns: simple pct
        def _ret(nh: int) -> float:
            if len(pts) <= nh:
                return 0.0
            p0 = pts[-nh-1]["price"]
            p1 = pts[-1]["price"]
            return (p1 - p0) / p0 if p0 else 0.0

        returns[coin] = {
            "h1": round(_ret(1), 6),
            "h24": round(_ret(24), 6),
            "h72": round(_ret(72 - 1) if len(pts) >= 72 else 0.0, 6),
        }

    payload = {
        "generated_at": _iso(now),
        "vs": vs,
        "coins": coins,
        "window_hours": window_h,
        "series": series,
        "returns": returns,
        "demo": True,  # local builder is always demo
        "attribution": "CoinGecko",
    }
    return payload, "[demo_fallback: local builder]"


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Market Context summary. IMPORTANT: demo-vs-live is controlled by MW_DEMO only.
    Global DEMO_MODE is intentionally ignored here.
    """
    artifacts = _artifacts_dir(ctx)

    market_demo = _env_bool("MW_DEMO", False)

    # Choose builder
    if _BUILDER is not None:
        try:
            payload, notes = _BUILDER(force_demo=market_demo)  # type: ignore[arg-type]
        except Exception:
            # If live builder throws during import-time tests, fallback to local
            payload, notes = _build_market_context_local(force_demo=True)
    else:
        payload, notes = _build_market_context_local(force_demo=True)

    # Persist JSON
    out_json = ctx.models_dir / "market_context.json"
    out_json.write_text(__import__("json").dumps(payload, ensure_ascii=False), encoding="utf-8")

    # Charts
    coins = payload.get("coins", [])
    series_by = payload.get("series", {}) or {}
    for coin in coins:
        s = series_by.get(coin) or []
        if s:
            xs = [datetime.fromtimestamp(pt["t"], tz=timezone.utc) for pt in s]
            rs = []
            for i in range(1, len(s)):
                p0, p1 = s[i - 1]["price"], s[i]["price"]
                try:
                    rs.append((p1 - p0) / p0 if p0 else 0.0)
                except Exception:
                    rs.append(0.0)
            _plot_price(s, title=f"Price Trend — {coin}", out=artifacts / f"market_trend_price_{coin}.png")
            _plot_returns(rs, xs[1:], title=f"Hourly Returns — {coin}",
                          out=artifacts / f"market_trend_returns_{coin}.png")

    # Markdown block
    lookback = int(payload.get("window_hours", 72))
    demo_tag = " (demo)" if payload.get("demo") else ""
    md.append(f"📈 Market Context (CoinGecko, {lookback}h){demo_tag}")

    series_by = payload.get("series", {}) or {}
    returns_by = payload.get("returns", {}) or {}
    last_prices = {c: (series_by.get(c) or [{}])[-1].get("price") for c in coins}

    for c in coins:
        spot = _fmt_money(last_prices.get(c))
        r = returns_by.get(c) or {}
        parts = [
            f"{c[:3].upper()} → {spot}",
            f"h1 {_fmt_pct(r.get('h1'))}",
            f"h24 {_fmt_pct(r.get('h24'))}",
            f"h72 {_fmt_pct(r.get('h72'))}",
        ]
        vol_hint = ""
        try:
            h1 = abs(float(r.get("h1", 0.0)))
            h24 = abs(float(r.get("h24", 0.0)))
            if h1 > 0.008 or h24 > 0.02:
                vol_hint = " [vol ↑]"
        except Exception:
            pass
        md.append(" • " + " | ".join(parts) + vol_hint)

    extra = " — Data via CoinGecko API; subject to plan rate limits."
    if notes:
        extra += f" {notes}"
    md.append(extra)