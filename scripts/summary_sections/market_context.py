# scripts/summary_sections/market_context.py
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import SummaryContext, _iso
from scripts.market.ingest_market import build_market_context  # live + demo fallback

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
    # Honor ARTIFACTS_DIR if set; else repo-root artifacts sibling to models/
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

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Market Context summary. IMPORTANT: demo-vs-live is controlled by MW_DEMO only.
    Global DEMO_MODE is intentionally ignored here so the rest of CI can stay demo.
    """
    artifacts = _artifacts_dir(ctx)

    # Only this knob decides demo vs live:
    market_demo = _env_bool("MW_DEMO", False)

    # Pull + build consolidated context (ingestor internally handles retries/backoff
    # and will return a demo payload if live fetch fails).
    payload, notes = build_market_context(force_demo=market_demo)

    # Persist JSON
    out_json = ctx.models_dir / "market_context.json"
    out_json.write_text(
        __import__("json").dumps(payload, ensure_ascii=False),
        encoding="utf-8"
    )

    # Charts
    coins = payload.get("coins", [])
    series_by = payload.get("series", {}) or {}
    returns_by = payload.get("returns", {}) or {}

    for coin in coins:
        s = series_by.get(coin) or []
        if s:
            xs = [datetime.fromtimestamp(pt["t"], tz=timezone.utc) for pt in s]
            rs = []
            for i in range(1, len(s)):
                p0, p1 = s[i-1]["price"], s[i]["price"]
                try:
                    rs.append((p1 - p0) / p0 if p0 else 0.0)
                except Exception:
                    rs.append(0.0)

            _plot_price(s, title=f"Price Trend — {coin}", out=artifacts / f"market_trend_price_{coin}.png")
            _plot_returns(rs, xs[1:], title=f"Hourly Returns — {coin}", out=artifacts / f"market_trend_returns_{coin}.png")

    # Markdown block
    lookback = int(payload.get("window_hours", 72))
    demo_tag = " (demo)" if payload.get("demo") else ""
    md.append(f"📈 Market Context (CoinGecko, {lookback}h){demo_tag}")

    vs = payload.get("vs", "usd")
    last_prices = {c: (series_by.get(c) or [{}])[-1].get("price") for c in coins}
    for c in coins:
        spot = _fmt_money(last_prices.get(c))
        r = returns_by.get(c) or {}
        parts = [f"{c[:3].upper()} → {spot}",
                 f"h1 {_fmt_pct(r.get('h1'))}",
                 f"h24 {_fmt_pct(r.get('h24'))}",
                 f"h72 {_fmt_pct(r.get('h72'))}"]
        # simple volatility hint
        vol_hint = ""
        try:
            h1 = float(r.get("h1", 0.0))
            h24 = abs(float(r.get("h24", 0.0)))
            if abs(h1) > 0.008 or h24 > 0.02:
                vol_hint = " [vol ↑]"
        except Exception:
            pass
        md.append(" • " + " | ".join(parts) + vol_hint)

    extra = " — Data via CoinGecko API; subject to plan rate limits."
    if notes:
        extra += f" {notes}"
    md.append(extra)