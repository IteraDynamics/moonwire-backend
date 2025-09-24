# scripts/summary_sections/market_context.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List

from .common import SummaryContext, ensure_dir, _iso  # assuming these exist in your repo
from scripts.market.ingest_market import build_market_context


def render_market_context(ctx: SummaryContext) -> str:
    """
    Renders the Market Context section.
    Always reads/writes models/market_context.json and emits plots into artifacts/.
    """
    models_dir = Path("models")
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    ensure_dir(models_dir)
    ensure_dir(artifacts_dir)

    coins = [c.strip() for c in os.getenv("MW_CG_COINS", "bitcoin,ethereum,solana").split(",") if c.strip()]
    vs = os.getenv("MW_CG_VS_CURRENCY", "usd").strip().lower()
    lookback_h = int(os.getenv("MW_CG_LOOKBACK_H", "72"))

    data = build_market_context(coins, vs, lookback_h, models_dir, artifacts_dir)

    # one-line headline for summary
    lines: List[str] = []
    tag = "(demo)" if data.get("demo") else ""
    if data.get("demo") and data.get("demo_reason"):
        tag = f"(demo) [{data['demo_reason']}]"
    lines.append(f"📈 Market Context (CoinGecko, {lookback_h}h) {tag}")

    # compact bullets: COIN → $price | h1 … | h24 … | h72 …
    def fmt_price(p: float) -> str:
        return f"${p:,.2f}"

    for c in coins:
        last_price = None
        s = data["series"].get(c, [])
        if s:
            last_price = s[-1]["price"]
        else:
            # fallback to simple/price if present in JSON (not stored by builder, so use series)
            last_price = 0.0
        r = data["returns"].get(c, {})
        h1 = f"{r.get('h1', 0.0):+0.1%}"
        h24 = f"{r.get('h24', 0.0):+0.1%}"
        h72 = f"{r.get('h72', 0.0):+0.1%}"
        lines.append(f"• {c.upper()} → {fmt_price(last_price or 0.0)} | h1 {h1} | h24 {h24} | h72 {h72}")

    # attribution
    lines.append("— Data via CoinGecko API; subject to plan rate limits.")
    return "\n".join(lines)