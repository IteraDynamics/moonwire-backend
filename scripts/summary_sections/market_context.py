# scripts/summary_sections/market_context.py
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict

from .common import SummaryContext, ensure_dir, _iso
from scripts.market.ingest_market import build_market_context  # live + demo fallback

def _fmt_money(p: float) -> str:
    try:
        return f"${float(p):,.2f}"
    except Exception:
        return "$—"

def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    try:
        pct = float(x) * 100.0
        sign = "+" if pct > 0 else ""
        return f"{sign}{pct:.1f}%"
    except Exception:
        return "—"

def _sym(coin_id: str) -> str:
    mapping = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL"}
    return mapping.get(coin_id.lower(), coin_id[:3].upper())

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build Market Context from CoinGecko (or deterministic demo if MW_DEMO=true or HTTP fails).
    Writes:
      - models/market_context.json
      - artifacts/market_trend_price_<coin>.png
      - artifacts/market_trend_returns_<coin>.png
    Appends a one-block markdown summary.
    """
    # ONLY this env controls demo/live. DO NOT derive from ctx.is_demo.
    mw_demo = os.getenv("MW_DEMO", "false").lower() in ("1", "true", "yes")

    lookback_h = int(os.getenv("MW_CG_LOOKBACK_H", "72"))
    coins_csv = os.getenv("MW_CG_COINS", "bitcoin,ethereum,solana")
    vs = os.getenv("MW_CG_VS_CURRENCY", "usd")
    coins = [c.strip().lower() for c in coins_csv.split(",") if c.strip()]

    # Where to place artifacts (respect ARTIFACTS_DIR if provided)
    artifacts_dir = os.getenv("ARTIFACTS_DIR")
    if artifacts_dir:
        artifacts_dir = Path(artifacts_dir)
    else:
        artifacts_dir = Path(ctx.models_dir).parent / "artifacts"
    ensure_dir(artifacts_dir)

    # Build (this handles live fetch + demo fallback internally)
    payload = build_market_context(
        models_dir=ctx.models_dir,
        logs_dir=ctx.logs_dir,
        artifacts_dir=artifacts_dir,
        coins=coins,
        vs=vs,
        lookback_h=lookback_h,
        force_demo=mw_demo,  # <-- the only demo switch
    )

    demo = bool(payload.get("demo"))
    series = payload.get("series", {})
    returns = payload.get("returns", {})

    hdr_demo = " (demo)" if demo else ""
    md.append(f"📈 Market Context (CoinGecko, {lookback_h}h){hdr_demo}")

    # One line per coin
    for cid in coins:
        sym = _sym(cid)
        # last price from series if present
        last_price = None
        pts = series.get(cid) or []
        if pts:
            last_price = pts[-1].get("price")
        r = returns.get(cid, {})
        h1 = _fmt_pct(r.get("h1"))
        h24 = _fmt_pct(r.get("h24"))
        h72 = _fmt_pct(r.get("h72"))

        # Simple “vol ↑” if absolute h72 > 5%
        vol_tag = " [vol ↑]" if (isinstance(r.get("h72"), (int, float)) and abs(float(r["h72"])) >= 0.05) else ""
        md.append(f"• {sym} → {_fmt_money(last_price)} | h1 {h1} | h24 {h24} | h72 {h72}{vol_tag}")

    md.append("— Data via CoinGecko API; subject to plan rate limits.")
    # If build_market_context flagged a local demo fallback due to HTTP failures, surface it
    if demo and payload.get("demo_reason"):
        md.append(f"[demo_fallback: {payload['demo_reason']}]")