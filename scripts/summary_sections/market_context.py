from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

from .common import SummaryContext, ensure_dir

# local import
from scripts.market.ingest_market import run_ingest


def _coin_symbol(coin_id: str) -> str:
    """
    Map common CoinGecko ids to canonical tickers, with a sane fallback.
    """
    if not coin_id:
        return "N/A"
    m = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "solana": "SOL",
        # extend as needed
    }
    if coin_id in m:
        return m[coin_id]
    sym = coin_id.split("-")[0].upper()
    return sym[:6] if len(sym) > 6 else sym


def append(md: List[str], ctx: SummaryContext) -> None:
    # decide artifact dirs similar to other sections
    artifact_dirs = [
        Path("artifacts"),
        ctx.logs_dir.parent / "artifacts",
        ctx.models_dir.parent / "artifacts",
    ]
    # unique dirs, left-most wins
    seen, arts = set(), []
    for d in artifact_dirs:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            arts.append(d)

    ensure_dir(ctx.logs_dir)
    ensure_dir(ctx.models_dir)
    for d in arts:
        ensure_dir(d)

    # invoke ingest (fail-soft to demo if needed)
    try:
        out = run_ingest(ctx.logs_dir, ctx.models_dir, arts[-1])
        demo = bool(out.get("demo"))
    except Exception:
        demo = True
        os.environ["MW_DEMO"] = "true"
        out = run_ingest(ctx.logs_dir, ctx.models_dir, arts[-1])

    vs = out.get("vs", "usd")
    look_h = int(out.get("window_hours", 72))
    coins = out.get("coins", [])
    rets: Dict[str, Dict[str, float]] = out.get("returns", {})
    series = out.get("series", {})

    # header
    md.append(f"📈 Market Context (CoinGecko, {look_h}h){' (demo)' if demo else ''}")
    # If you want an explicit mode line, uncomment the next line:
    # md.append(f"mode: {'demo' if demo else 'real'} • vs={vs}")

    def _fmt_price(p):
        try:
            return f"${float(p):,.2f}"
        except Exception:
            return str(p)

    # one line per coin
    for cid in coins:
        s = series.get(cid) or []
        last = s[-1]["price"] if s else None
        r = rets.get(cid, {})
        h1 = r.get("h1", 0.0)
        h24 = r.get("h24", 0.0)
        h72 = r.get("h72", 0.0)
        vol = " [vol ↑]" if abs(h24) > 0.03 else ""
        sym = _coin_symbol(cid)
        md.append(f"{sym:<4}→ {_fmt_price(last)} | h1 {h1:+.1%} | h24 {h24:+.1%} | h72 {h72:+.1%}{vol}")

    # attribution
    md.append("— Data via CoinGecko API; subject to plan rate limits.")