# scripts/summary_sections/market_context.py
from __future__ import annotations

import os
import json
import math
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

from .common import SummaryContext, ensure_dir, _iso

# ---- tiny format helpers ----
def _fmt_money(p: float | None) -> str:
    try:
        return f"${float(p):,.2f}"
    except Exception:
        return "$—"

def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    try:
        v = float(x) * 100.0
        sign = "+" if v > 0 else ""
        return f"{sign}{v:.1f}%"
    except Exception:
        return "—"

def _sym(coin_id: str) -> str:
    mapping = {"bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL"}
    return mapping.get(coin_id.lower(), coin_id[:3].upper())


# ---- demo builder (used if ingest module missing or fetch fails) ----
def _build_demo_market_context(
    models_dir: Path,
    logs_dir: Path,
    artifacts_dir: Path,
    coins: List[str],
    vs: str,
    lookback_h: int,
    reason: str = "local demo",
) -> Dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None  # tolerate headless import failures

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(hours=lookback_h)

    # deterministic seed so CI is stable
    random.seed(1337)
    series: Dict[str, List[Dict[str, float]]] = {}
    returns: Dict[str, Dict[str, float]] = {}

    for cid in coins:
        pts: List[Dict[str, float]] = []
        # pick a base + amplitude by coin
        if cid == "bitcoin":
            base, amp = 60000.0, 4000.0
        elif cid == "ethereum":
            base, amp = 3000.0, 200.0
        else:
            base, amp = 150.0, 12.0

        t = start
        k = 2.0 * math.pi / 24.0  # ~daily-ish wiggle
        i = 0
        while t <= now:
            noise = random.uniform(-0.6, 0.6)
            price = base + amp * math.sin(k * i) + noise * (0.02 * base)
            pts.append({"t": int(t.timestamp()), "price": float(max(0.01, price))})
            t += timedelta(hours=1)
            i += 1
        series[cid] = pts

        # compute simple returns h1/h24/h72 off last price
        def _ret(hours: int) -> float | None:
            if not pts:
                return None
            last = pts[-1]["price"]
            t0 = now - timedelta(hours=hours)
            # find nearest bucket at/before t0
            prior = None
            for p in reversed(pts):
                if p["t"] <= int(t0.timestamp()):
                    prior = p["price"]
                    break
            if prior is None or prior == 0:
                return None
            return (last - prior) / prior

        returns[cid] = {"h1": _ret(1), "h24": _ret(24), "h72": _ret(72)}

        # append one line to market log
        ensure_dir(logs_dir)
        logp = logs_dir / "market_prices.jsonl"
        latest_price = pts[-1]["price"] if pts else None
        log_line = {
            "ts_utc": _iso(now),
            "id": cid,
            "symbol": _sym(cid).lower(),
            "vs": vs,
            "price": latest_price,
            "source": "coingecko",
            "attribution": "CoinGecko",
            "demo": True,
        }
        with logp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_line) + "\n")

        # plots
        if plt and pts:
            times = [datetime.fromtimestamp(p["t"], tz=timezone.utc) for p in pts]
            prices = [p["price"] for p in pts]

            # price line
            try:
                plt.figure(figsize=(7, 3))
                plt.plot(times, prices, marker="o", linewidth=1)
                plt.title(f"Price — {cid} ({lookback_h}h) [demo]")
                plt.xlabel("time")
                plt.ylabel(f"price ({vs})")
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                ensure_dir(artifacts_dir)
                plt.savefig(artifacts_dir / f"market_trend_price_{cid}.png", dpi=120)
                plt.close()
            except Exception:
                pass

            # hourly simple returns bar
            rets = []
            for j in range(1, len(prices)):
                a, b = prices[j - 1], prices[j]
                rets.append(0.0 if a == 0 else (b - a) / a)
            try:
                plt.figure(figsize=(7, 3))
                plt.bar(range(len(rets)), rets)
                plt.axhline(0.0, linewidth=1)
                plt.title(f"Hourly returns — {cid} ({lookback_h}h) [demo]")
                plt.xlabel("hour index")
                plt.ylabel("return")
                plt.tight_layout()
                plt.savefig(artifacts_dir / f"market_trend_returns_{cid}.png", dpi=120)
                plt.close()
            except Exception:
                pass

    payload = {
        "generated_at": _iso(now),
        "vs": vs,
        "coins": coins,
        "window_hours": lookback_h,
        "series": series,
        "returns": returns,
        "demo": True,
        "demo_reason": reason,
        "attribution": "CoinGecko",
    }

    ensure_dir(models_dir)
    (models_dir / "market_context.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build Market Context from CoinGecko (live when available) or deterministic demo (when forced or on failure).
    Writes:
      - models/market_context.json
      - artifacts/market_trend_price_<coin>.png
      - artifacts/market_trend_returns_<coin>.png
    Appends a concise markdown block to `md`.
    """
    # This flag alone controls demo vs live preference. We DO NOT inherit ctx.is_demo here.
    mw_demo = os.getenv("MW_DEMO", "false").lower() in ("1", "true", "yes")

    lookback_h = int(os.getenv("MW_CG_LOOKBACK_H", "72"))
    coins_csv = os.getenv("MW_CG_COINS", "bitcoin,ethereum,solana")
    vs = os.getenv("MW_CG_VS_CURRENCY", "usd")
    coins = [c.strip().lower() for c in coins_csv.split(",") if c.strip()]

    # Artifact dir (respect ARTIFACTS_DIR if provided)
    art_dir_env = os.getenv("ARTIFACTS_DIR")
    artifacts_dir = Path(art_dir_env) if art_dir_env else Path(ctx.models_dir).parent / "artifacts"
    ensure_dir(artifacts_dir)

    # Try live builder via lazy import; on any error, fall back to demo.
    payload: Dict[str, Any] | None = None
    demo_reason: str | None = None

    if not mw_demo:
        try:
            # Lazy import so that tests collecting modules don't break if ingest package is absent.
            from scripts.market.ingest_market import build_market_context  # type: ignore

            payload = build_market_context(
                models_dir=ctx.models_dir,
                logs_dir=ctx.logs_dir,
                artifacts_dir=artifacts_dir,
                coins=coins,
                vs=vs,
                lookback_h=lookback_h,
                force_demo=False,  # live path preferred
            )
        except Exception as e:
            demo_reason = f"live_fetch_failed: {e.__class__.__name__}"

    if payload is None:
        payload = _build_demo_market_context(
            models_dir=ctx.models_dir,
            logs_dir=ctx.logs_dir,
            artifacts_dir=artifacts_dir,
            coins=coins,
            vs=vs,
            lookback_h=lookback_h,
            reason=demo_reason or ("forced_demo" if mw_demo else "missing_ingest_module"),
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

        # Simple “vol ↑” if |h72| >= 5%
        vol_tag = " [vol ↑]" if (isinstance(r.get("h72"), (int, float)) and abs(float(r["h72"])) >= 0.05) else ""
        md.append(f"• {sym} → {_fmt_money(last_price)} | h1 {h1} | h24 {h24} | h72 {h72}{vol_tag}")

    md.append("— Data via CoinGecko API; subject to plan rate limits.")
    if demo and payload.get("demo_reason"):
        md.append(f"[demo_fallback: {payload['demo_reason']}]")