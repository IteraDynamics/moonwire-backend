# scripts/summary_sections/market_context.py
from __future__ import annotations

import os
from typing import Dict, List, Any

from .common import SummaryContext, _write_json  # ensure_dir implied via _write_json
from scripts.market.ingest_market import build_market_context


def _fmt_money(v: float, vs: str) -> str:
    """Format price with a leading currency sign when vs=='usd', else '123.45 vs'."""
    if v is None:
        return "—"
    if vs.lower() == "usd":
        return f"${float(v):,.2f}"
    return f"{float(v):,.2f} {vs.lower()}"


def _pct_str(x: float) -> str:
    try:
        return f"{x*100:+.1f}%"
    except Exception:
        return "—"


def _ticker_from_coin(coin: str) -> str:
    mapping = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "solana": "SOL",
    }
    return mapping.get(coin.lower().strip(), coin.upper())


def _summarize(ctx_data: Dict[str, Any]) -> List[str]:
    """
    Build short markdown lines summarizing the market context.
    Example:
      📈 Market Context (CoinGecko, 72h) (demo)
      • BTC → $65,100.14 | h1 +1.0% | h24 -2.0% | h72 +8.7%
      — Data via CoinGecko API; subject to plan rate limits.
    """
    lines: List[str] = []

    hours = ctx_data.get("window_hours") or ctx_data.get("lookback_h") or 72
    demo = bool(ctx_data.get("demo"))
    vs = (ctx_data.get("vs") or "usd").lower()
    returns: Dict[str, Dict[str, float]] = ctx_data.get("returns") or {}
    series: Dict[str, List[Dict[str, Any]]] = ctx_data.get("series") or {}
    coins: List[str] = ctx_data.get("coins") or []

    title = f"📈 Market Context (CoinGecko, {hours}h)"
    if demo:
        title += " (demo)"
    lines.append(title)

    # One bullet per coin
    for coin in coins:
        arr = series.get(coin) or []
        last_px = (arr[-1]["price"] if arr else None)
        r = returns.get(coin) or {}
        h1 = _pct_str(r.get("h1", 0.0))
        h24 = _pct_str(r.get("h24", 0.0))
        h72 = _pct_str(r.get("h72", 0.0))
        ticker = _ticker_from_coin(coin)
        money = _fmt_money(last_px, vs)
        lines.append(f"• {ticker} → {money} | h1 {h1} | h24 {h24} | h72 {h72}")

    # Attribution + reason if demo
    attr = "— Data via CoinGecko API; subject to plan rate limits."
    demo_reason = ctx_data.get("demo_reason")
    if demo and demo_reason:
        attr += f" [{demo_reason}]"
    lines.append(attr)

    return lines


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build market context (live or demo) and append a short markdown section
    to the provided md list. Also writes models/market_context.json so the
    rest of the pipeline (and tests) can find it.
    """
    # Respect MW_DEMO environment variable as the ingest does
    # (build_market_context reads MW_DEMO / MW_CG_* knobs internally).
    ctx_data = build_market_context()

    # Persist JSON artifact for tests and downstream steps
    out_json = ctx.models_dir / "market_context.json"
    _write_json(out_json, ctx_data)

    # Append markdown lines
    md.extend(_summarize(ctx_data))