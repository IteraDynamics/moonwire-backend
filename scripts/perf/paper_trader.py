# scripts/perf/paper_trader.py
from __future__ import annotations
import json, os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math
import numpy as np

from scripts.summary_sections.common import ensure_dir, _iso
from .performance_metrics import compute_metrics

@dataclass
class Ctx:
    logs_dir: Path
    models_dir: Path
    artifacts_dir: Path
    is_demo: bool = False

def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _load_prices(models_dir: Path, symbols: List[str], lookback_h: int) -> Dict[str, List[Dict[str, Any]]]:
    j = models_dir / "market_context.json"
    if j.exists():
        try:
            data = json.loads(j.read_text())
            ts_cut = _now() - timedelta(hours=lookback_h)
            out: Dict[str, List[Dict[str, Any]]] = {}
            for sym in symbols:
                series = data.get(sym.lower()) or data.get(sym.upper()) or data.get(sym)
                if not series:
                    continue
                filtered = [row for row in series if datetime.fromisoformat(row["ts"].replace("Z","+00:00")) >= ts_cut]
                out[sym] = filtered
            if out:
                return out
        except Exception:
            pass
    # demo fallback: synth series hourly
    out: Dict[str, List[Dict[str, Any]]] = {}
    start = _now() - timedelta(hours=lookback_h)
    steps = lookback_h
    rng = np.random.default_rng(42)
    for sym in symbols:
        px = 100.0 + rng.normal(0, 1, size=steps).cumsum()
        out[sym] = [{"ts": _iso(start + timedelta(hours=i)), "price": float(px[i])} for i in range(steps)]
    return out

def _load_signals(logs_dir: Path, symbols: List[str], lookback_h: int) -> List[Dict[str, Any]]:
    f = logs_dir / "signals.jsonl"
    if f.exists():
        try:
            lines = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
            ts_cut = _now() - timedelta(hours=lookback_h)
            keep = []
            for r in lines:
                ts = datetime.fromisoformat(str(r["ts"]).replace("Z","+00:00"))
                if ts >= ts_cut and str(r.get("symbol","")).upper() in symbols:
                    keep.append(r)
            keep.sort(key=lambda r: r["ts"])
            if keep:
                return keep
        except Exception:
            pass
    # demo fallback: synth few signals
    base = _now() - timedelta(hours=min(lookback_h, 12))
    synth = []
    for i, sym in enumerate(symbols):
        for k in range(3):
            ts = base + timedelta(hours=k*2 + i)
            synth.append({
                "id": f"demo_{sym}_{k}",
                "ts": _iso(ts),
                "symbol": sym,
                "direction": "buy" if k % 2 == 0 else "sell",
                "confidence": 0.7,
            })
    synth.sort(key=lambda r: r["ts"])
    return synth

def _match_price_at(prices: List[Dict[str, Any]], ts: datetime) -> float:
    # next available sample at or after ts
    for row in prices:
        trow = datetime.fromisoformat(row["ts"].replace("Z","+00:00"))
        if trow >= ts:
            return float(row["price"])
    return float(prices[-1]["price"])

def _plot_png(path: Path, xs: List[datetime], ys: List[float], title: str):
    ensure_dir(path.parent)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6,3), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(xs, ys)
        ax.set_title(title)
        ax.grid(True, linewidth=0.4)
        fig.tight_layout()
        fig.savefig(str(path))
        plt.close(fig)
    except Exception:
        # write 1x1 placeholder
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
            b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
        )

def run_paper_trader(ctx, mode: str = "backtest") -> Dict[str, Any]:
    # ctx is compatible with your SummaryContext; we only need dirs + is_demo
    symbols = [s.strip().upper() for s in os.getenv("MW_PERF_SYMBOLS","BTC,ETH,SOL").split(",") if s.strip()]
    lookback_h = int(os.getenv("MW_PERF_LOOKBACK_H","72"))
    horizon_min = int(os.getenv("MW_PERF_HORIZON_MIN","60"))
    slip_bps = float(os.getenv("MW_PERF_SLIPPAGE_BPS","2"))
    fee_bps = float(os.getenv("MW_PERF_FEES_BPS","1"))
    capital = float(os.getenv("MW_PERF_CAPITAL","100000"))
    rf = float(os.getenv("MW_PERF_RISK_FREE","0.0"))

    prices_by_sym = _load_prices(ctx.models_dir, symbols, lookback_h)
    signals = _load_signals(ctx.logs_dir, symbols, lookback_h)

    # basic 1x notion per trade, single-position per symbol model
    trades: List[Dict[str, Any]] = []
    equity_log: List[Dict[str, Any]] = []
    eq = capital
    last_ts = _now()

    open_pos: Dict[str, Dict[str, Any]] = {}  # sym -> pos dict

    for sig in signals:
        sym = sig["symbol"].upper()
        ts = datetime.fromisoformat(sig["ts"].replace("Z","+00:00"))
        last_ts = max(last_ts, ts)
        px_series = prices_by_sym.get(sym)
        if not px_series:
            continue
        side = 1 if sig.get("direction","buy").lower() == "buy" else -1
        entry_px = _match_price_at(px_series, ts)
        entry_px *= (1 + side * (slip_bps/1e4))  # slippage

        # Close any existing position if reverse or time horizon hit
        pos = open_pos.get(sym)
        if pos and (pos["side"] != side or ts >= pos["entry_ts"] + timedelta(minutes=horizon_min)):
            exit_px = _match_price_at(px_series, ts)
            exit_px *= (1 - pos["side"] * (slip_bps/1e4))
            fees = (abs(pos["entry_px"]) + abs(exit_px)) * (fee_bps/1e4)
            pnl = (exit_px - pos["entry_px"]) * pos["side"] - fees
            eq += pnl
            trades.append({
                "ts_close": sig["ts"],
                "symbol": sym,
                "side": "long" if pos["side"]==1 else "short",
                "entry": pos["entry_px"],
                "exit": exit_px,
                "pnl": pnl,
                "pnl_pct": pnl / pos["entry_px"] if pos["entry_px"] else 0.0,
            })
            open_pos.pop(sym, None)
            equity_log.append({"ts": sig["ts"], "equity": eq})

        # Open/flip into new position
        open_pos[sym] = {"side": side, "entry_px": entry_px, "entry_ts": ts}

    # time-based close for any leftover positions
    if signals:
        cutoff = last_ts + timedelta(minutes=horizon_min)
        for sym, pos in list(open_pos.items()):
            px_series = prices_by_sym.get(sym)
            exit_px = _match_price_at(px_series, cutoff)
            exit_px *= (1 - pos["side"] * (slip_bps/1e4))
            fees = (abs(pos["entry_px"]) + abs(exit_px)) * (fee_bps/1e4)
            pnl = (exit_px - pos["entry_px"]) * pos["side"] - fees
            eq += pnl
            trades.append({
                "ts_close": _iso(cutoff),
                "symbol": sym,
                "side": "long" if pos["side"]==1 else "short",
                "entry": pos["entry_px"],
                "exit": exit_px,
                "pnl": pnl,
                "pnl_pct": pnl / pos["entry_px"] if pos["entry_px"] else 0.0,
            })
            equity_log.append({"ts": _iso(cutoff), "equity": eq})

    # derive returns from equity log
    equity_vals = [capital] + [row["equity"] for row in equity_log]
    returns = []
    for i in range(1, len(equity_vals)):
        prev = equity_vals[i-1]
        cur = equity_vals[i]
        returns.append((cur - prev) / prev if prev else 0.0)

    hours_span = float(lookback_h)
    agg = compute_metrics(equity_vals, returns, trades, risk_free=rf, hours_span=hours_span)

    # per-symbol quick tally
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        t = [tr for tr in trades if tr["symbol"] == sym]
        if not t:
            continue
        eq_sym = [capital]  # simple — reuse capital baseline (symbol slices are illustrative)
        cur = capital
        for tr in t:
            cur += tr["pnl"]
            eq_sym.append(cur)
        rets_sym = []
        for i in range(1, len(eq_sym)):
            rets_sym.append((eq_sym[i]-eq_sym[i-1]) / eq_sym[i-1] if eq_sym[i-1] else 0.0)
        by_symbol[sym] = {
            "trades": len(t),
            "sharpe": agg["sharpe"] if rets_sym else 0.0,
            "sortino": agg["sortino"] if rets_sym else 0.0,
            "max_drawdown": agg["max_drawdown"],
            "win_rate": sum(1 for x in t if x["pnl"]>0)/len(t),
            "profit_factor": (sum(x["pnl"] for x in t if x["pnl"]>0) /
                              (-sum(x["pnl"] for x in t if x["pnl"]<0)) if any(x["pnl"]<0 for x in t) else float("inf")),
            "cagr": None,
        }

    result = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mode": os.getenv("MW_PERF_MODE","backtest"),
        "window_hours": lookback_h,
        "capital": capital,
        "by_symbol": by_symbol,
        "aggregate": {
            "trades": agg["trades"],
            "sharpe": agg["sharpe"],
            "sortino": agg["sortino"],
            "max_drawdown": agg["max_drawdown"],
            "win_rate": agg["win_rate"],
            "profit_factor": agg["profit_factor"],
        },
        "demo": not (Path(ctx.logs_dir/"signals.jsonl").exists() and Path(ctx.models_dir/"market_context.json").exists()),
    }

    # write logs/artifacts
    ensure_dir(ctx.logs_dir); ensure_dir(ctx.artifacts_dir); ensure_dir(ctx.models_dir)
    # logs
    (ctx.logs_dir / "trades.jsonl").write_text("\n".join(json.dumps(t) for t in trades) + ("\n" if trades else ""))
    (ctx.logs_dir / "equity_curve.jsonl").write_text("\n".join(json.dumps(r) for r in equity_log) + ("\n" if equity_log else ""))

    # charts
    xs = [datetime.fromisoformat(row["ts"].replace("Z","+00:00")) for row in equity_log] or [datetime.now(timezone.utc)]
    ys = [row["equity"] for row in equity_log] or [capital]
    _plot_png(ctx.artifacts_dir / "perf_equity_curve.png", xs, ys, "Equity Curve")
    # drawdown series
    dd = []
    peak = -1e18
    for v in ys:
        peak = max(peak, v)
        dd.append((v - peak)/peak if peak else 0.0)
    _plot_png(ctx.artifacts_dir / "perf_drawdown.png", xs, dd, "Drawdown")

    # returns hist
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.figure(figsize=(6,3), dpi=100)
        ax = fig.add_subplot(111)
        ax.hist(np.asarray(returns, dtype=float), bins=20)
        ax.set_title("Returns Histogram")
        fig.tight_layout()
        fig.savefig(str(ctx.artifacts_dir / "perf_returns_hist.png"))
        plt.close(fig)
    except Exception:
        (ctx.artifacts_dir / "perf_returns_hist.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82")

    # by-symbol bar (using total PnL)
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        labels = list(by_symbol.keys()) or ["NA"]
        vals = [sum(t["pnl"] for t in trades if t["symbol"]==s) for s in labels] if by_symbol else [0.0]
        fig = plt.figure(figsize=(6,3), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(labels, vals)
        ax.set_title("PnL by Symbol")
        fig.tight_layout()
        fig.savefig(str(ctx.artifacts_dir / "perf_by_symbol_bar.png"))
        plt.close(fig)
    except Exception:
        (ctx.artifacts_dir / "perf_by_symbol_bar.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82")

    # metrics json
    (ctx.models_dir / "performance_metrics.json").write_text(json.dumps(result, indent=2))

    return result