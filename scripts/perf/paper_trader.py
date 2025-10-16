# scripts/perf/paper_trader.py
from __future__ import annotations
import json, os, math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from scripts.summary_sections.common import ensure_dir, _iso
from .performance_metrics import compute_metrics, EPS


@dataclass
class Ctx:
    logs_dir: Path
    models_dir: Path
    artifacts_dir: Path
    is_demo: bool = False


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _load_prices(models_dir: Path, symbols: List[str], lookback_h: int) -> Dict[str, List[Dict[str, Any]]]:
    f = models_dir / "market_context.json"
    if f.exists():
        try:
            data = json.loads(f.read_text())
            out = {}
            cutoff = _now() - timedelta(hours=lookback_h)
            for sym in symbols:
                series = data.get(sym.lower()) or data.get(sym.upper())
                if not series:
                    continue
                filtered = [r for r in series if datetime.fromisoformat(r["ts"].replace("Z", "+00:00")) >= cutoff]
                if filtered:
                    out[sym] = filtered
            if out:
                return out
        except Exception:
            pass

    rng = np.random.default_rng(42)
    out = {}
    start = _now() - timedelta(hours=lookback_h)
    for sym in symbols:
        vals = 100 + rng.normal(0, 1, size=lookback_h).cumsum()
        out[sym] = [{"ts": _iso(start + timedelta(hours=i)), "price": float(vals[i])} for i in range(lookback_h)]
    return out


def _load_signals(logs_dir: Path, symbols: List[str], lookback_h: int) -> List[Dict[str, Any]]:
    f = logs_dir / "signals.jsonl"
    if f.exists():
        try:
            lines = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            cutoff = _now() - timedelta(hours=lookback_h)
            out = []
            for r in lines:
                ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
                if ts >= cutoff and r.get("symbol", "").upper() in symbols:
                    out.append(r)
            out.sort(key=lambda r: r["ts"])
            if out:
                return out
        except Exception:
            pass

    # Demo fallback
    synth = []
    base = _now() - timedelta(hours=6)
    for i, sym in enumerate(symbols):
        for k in range(3):
            ts = base + timedelta(hours=k * 2 + i)
            synth.append(
                {
                    "id": f"demo_{sym}_{k}",
                    "ts": _iso(ts),
                    "symbol": sym,
                    "direction": "buy" if k % 2 == 0 else "sell",
                    "confidence": 0.7,
                }
            )
    synth.sort(key=lambda r: r["ts"])
    return synth


def _match_price(prices: List[Dict[str, Any]], ts: datetime) -> float:
    for row in prices:
        t = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
        if t >= ts:
            return float(row["price"])
    return float(prices[-1]["price"])


def _plot_png(path: Path, xs: List[datetime], ys: List[float], title: str):
    ensure_dir(path.parent)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(xs, ys)
        ax.set_title(title)
        ax.grid(True, lw=0.4)
        fig.tight_layout()
        fig.savefig(str(path))
        plt.close(fig)
    except Exception:
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
            b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
        )


def run_paper_trader(ctx, mode: str = "backtest") -> Dict[str, Any]:
    symbols = [s.strip().upper() for s in os.getenv("MW_PERF_SYMBOLS", "BTC,ETH,SOL").split(",")]
    lookback_h = int(os.getenv("MW_PERF_LOOKBACK_H", "72"))
    horizon_min = int(os.getenv("MW_PERF_HORIZON_MIN", "120"))
    slippage_bps = float(os.getenv("MW_PERF_SLIPPAGE_BPS", "1"))
    fees_bps = float(os.getenv("MW_PERF_FEES_BPS", "0.5"))
    capital = float(os.getenv("MW_PERF_CAPITAL", "100000"))

    prices = _load_prices(ctx.models_dir, symbols, lookback_h)
    signals = _load_signals(ctx.logs_dir, symbols, lookback_h)

    trades, eq_curve = [], []
    eq = capital
    open_pos = {}

    for sig in signals:
        sym = sig["symbol"].upper()
        pxs = prices.get(sym)
        if not pxs:
            continue
        ts = datetime.fromisoformat(sig["ts"].replace("Z", "+00:00"))
        side = "long" if sig["direction"].lower() == "buy" else "short"
        mult = 1 if side == "long" else -1
        entry = _match_price(pxs, ts) * (1 + mult * slippage_bps / 1e4)

        pos = open_pos.get(sym)
        if pos and (pos["side"] != side or ts >= pos["entry_ts"] + timedelta(minutes=horizon_min)):
            exit_px = _match_price(pxs, ts) * (1 - pos["mult"] * slippage_bps / 1e4)
            fees = (entry + exit_px) * (fees_bps / 1e4)
            pnl_abs = (exit_px - pos["entry_px"]) * pos["mult"] - fees
            pnl_frac = pnl_abs / max(pos["entry_px"], EPS)
            eq += capital * pnl_frac
            trades.append(
                {
                    "ts": _iso(ts),
                    "symbol": sym,
                    "side": pos["side"],
                    "entry": pos["entry_px"],
                    "exit": exit_px,
                    "pnl": pnl_abs,
                    "pnl_frac": pnl_frac,
                }
            )
            eq_curve.append({"ts": _iso(ts), "equity": eq})
            open_pos.pop(sym, None)

        open_pos[sym] = {"side": side, "entry_px": entry, "entry_ts": ts, "mult": mult}

    # Close leftovers
    cutoff = _now() + timedelta(minutes=horizon_min)
    for sym, pos in list(open_pos.items()):
        pxs = prices.get(sym)
        exit_px = _match_price(pxs, cutoff)
        fees = (exit_px + pos["entry_px"]) * (fees_bps / 1e4)
        pnl_abs = (exit_px - pos["entry_px"]) * pos["mult"] - fees
        pnl_frac = pnl_abs / max(pos["entry_px"], EPS)
        eq += capital * pnl_frac
        trades.append(
            {
                "ts": _iso(cutoff),
                "symbol": sym,
                "side": pos["side"],
                "entry": pos["entry_px"],
                "exit": exit_px,
                "pnl": pnl_abs,
                "pnl_frac": pnl_frac,
            }
        )
        eq_curve.append({"ts": _iso(cutoff), "equity": eq})

    # Metrics
    metrics = compute_metrics(eq_curve, trades)

    # Write logs & artifacts
    ensure_dir(ctx.logs_dir)
    ensure_dir(ctx.artifacts_dir)
    ensure_dir(ctx.models_dir)

    (ctx.logs_dir / "trades.jsonl").write_text("\n".join(json.dumps(t) for t in trades))
    (ctx.logs_dir / "equity_curve.jsonl").write_text("\n".join(json.dumps(r) for r in eq_curve))
    (ctx.models_dir / "performance_metrics.json").write_text(json.dumps(metrics, indent=2))

    xs = [datetime.fromisoformat(r["ts"].replace("Z", "+00:00")) for r in eq_curve]
    ys = [r["equity"] for r in eq_curve]
    _plot_png(ctx.artifacts_dir / "perf_equity_curve.png", xs, ys, "Equity Curve")

    dd = []
    peak = -1e9
    for v in ys:
        peak = max(peak, v)
        dd.append((v - peak) / max(peak, EPS))
    _plot_png(ctx.artifacts_dir / "perf_drawdown.png", xs, dd, "Drawdown")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.hist([(t["pnl_frac"]) for t in trades], bins=10)
    ax.set_title("Returns Histogram")
    fig.tight_layout()
    fig.savefig(ctx.artifacts_dir / "perf_returns_hist.png")
    plt.close(fig)

    labels = [s for s in symbols]
    vals = [sum(t["pnl"] for t in trades if t["symbol"] == s) for s in symbols]
    fig = plt.figure(figsize=(6, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(labels, vals)
    ax.set_title("PnL by Symbol")
    fig.tight_layout()
    fig.savefig(ctx.artifacts_dir / "perf_by_symbol_bar.png")
    plt.close(fig)

    return metrics