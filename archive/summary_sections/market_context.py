# scripts/summary_sections/market_context.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# We only import ingest functions at runtime (inside append) so that
# a bad import won't poison the module and break the guarded import in __init__.


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _fmt_pct(x: Optional[float]) -> str:
    try:
        return f"{x:+.1%}"
    except Exception:
        return "n/a"


def _fmt_price(x: Optional[float]) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return "n/a"


def _last_price(series: List[Dict[str, Any]]) -> Optional[float]:
    if not series:
        return None
    try:
        return float(series[-1]["price"])
    except Exception:
        return None


def _summarize_market(mc: Dict[str, Any]) -> List[str]:
    vs = mc.get("vs", "usd").upper()
    coins = mc.get("coins", [])
    window_h = mc.get("window_hours", 72)
    lines: List[str] = [f"📈 Market Context (CoinGecko, {window_h}h)"]

    series = mc.get("series", {})
    returns = mc.get("returns", {})

    for coin in coins:
        ser = series.get(coin, [])
        px = _last_price(ser)
        r = returns.get(coin, {})
        h1 = r.get("h1")
        h24 = r.get("h24")
        h72 = r.get("h72")
        lines.append(
            f"• {coin.upper()} → {_fmt_price(px)} | h1 {_fmt_pct(h1)} | h24 {_fmt_pct(h24)} | h72 {_fmt_pct(h72)}"
        )
    lines.append("— Data via CoinGecko API; subject to plan rate limits.")
    return lines


def append(md: List[str], ctx) -> None:
    """
    Build/refresh market context artifacts (demo-friendly) and append a short summary block.
    This function never raises; it writes a helpful message to md on failure.
    """
    # Resolve dirs from ctx (SummaryContext-compatible)
    try:
        logs_dir = Path(getattr(ctx, "logs_dir", Path("logs")))
        models_dir = Path(getattr(ctx, "models_dir", Path("models")))
        artifacts_dir = Path(getattr(ctx, "artifacts_dir", Path("artifacts")))
    except Exception:
        md.append("> ⚠️ Market Context: invalid context paths.\n")
        return

    # Ensure dirs exist
    for d in (logs_dir, models_dir, artifacts_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Try to run ingest (demo or live depending on env)
    try:
        from scripts.market.ingest_market import run_ingest, IngestPaths  # local import to avoid import-time failure

        paths = IngestPaths(logs_dir=logs_dir, models_dir=models_dir, artifacts_dir=artifacts_dir)
        run_ingest(paths=paths)
    except Exception as e:
        # Even if ingest fails, try to summarize whatever (if anything) is already on disk.
        md.append(f"> ⚠️ Market Context ingest failed: `{type(e).__name__}: {e}`\n")

    # Load artifact (created either now or previously)
    mc_path = models_dir / "market_context.json"
    mc = _read_json(mc_path)
    if not mc:
        md.append("> ⚠️ Market Context: artifact missing.\n")
        return

    # Compose markdown block
    demo_flag = mc.get("demo")
    demo_suffix = " (demo)" if demo_flag else ""
    md.append(f"### MoonWire CI Demo Summary{'' if os.getenv('GITHUB_ACTIONS') else ''}")
    lines = _summarize_market(mc)
    if lines:
        lines[0] = lines[0] + demo_suffix
    md.extend(lines)
    md.append("")  # trailing newline for spacing


__all__ = ["append"]