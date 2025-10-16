# scripts/perf/paper_trader.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Reuse shared helpers
from scripts.summary_sections.common import ensure_dir, _iso
from .performance_metrics import compute_metrics  # local metrics lib


# ---------------------------
# Small context shim (compatible with SummaryContext)
# ---------------------------

@dataclass
class _CtxShim:
    logs_dir: Path
    models_dir: Path
    artifacts_dir: Path
    is_demo: bool = False


# ---------------------------
# Utilities
# ---------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_dt(ts: str) -> datetime:
    # Accept "...Z" or ISO with offset
    if ts.endswith("Z"):
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return datetime.fromisoformat(ts)


def _nearest_price_after(prices: List[Dict[str, Any]], at: datetime) -> Tuple[datetime, float]:
    # prices sorted by ts; choose first with ts >= at
    for p in prices:
        t = _to_dt(p["ts"])
        if t >= at:
            return t, float(p["price"])
    # fallback to last
    t = _to_dt(prices[-1]["ts"])
    return t, float(prices[-1]["price"])


def _nearest_price_at_or_before(prices: List[Dict[str, Any]], at: datetime) -> Tuple[datetime, float]:
    prev = None
    for p in prices:
        t = _to_dt(p["ts"])
        if t > at:
            break
        prev = (t, float(p["price"]))
    if prev is not None:
        return prev
    # earliest
    t = _to_dt(prices[0]["ts"])
    return t, float(prices[0]["price"])


def _bps(x: float) -> float:
    return x / 10_000.0


# ---------------------------
# IO: load signals / prices (with demo fallbacks)
# ---------------------------

def _load_signals(logs_dir: Path, lookback_h: int, symbols: List[str]) -> List[Dict[str, Any]]:
    spath = logs_dir / "signals.jsonl"
    out: List[Dict[str, Any]] = []
    if spath.exists():
        for line in spath.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "symbol" in row and row.get("symbol") in symbols:
                out.append(row)
    if out:
        out.sort(key=lambda r: r.get("ts", ""))
        # restrict to lookback window
        cutoff = _now_utc() - timedelta(hours=lookback_h)
        out = [r for r in out if _to_dt(r["ts"]) >= cutoff]
        return out

    # Demo fallback: synthesize 3 signals per symbol over the window
    demo: List[Dict[str, Any]] = []
    t0 = _now_utc() - timedelta(hours=lookback_h)
    step = timedelta(hours=lookback_h // 4 or 1)
    for s in symbols:
        for i in range(3):
            ts = _iso(t0 + step * (i + 1))
            demo.append({
                "id": f"demo_{s}_{i}",
                "ts": ts,
                "symbol": s,
                "direction": "long" if i % 2 == 0 else "short",
                "confidence": 0.6 + 0.1 * (i % 2),
                "price": None,
            })
    demo.sort(key=lambda r: r["ts"])
    return demo


def _load_prices(models_dir: Path, lookback_h: int, symbols: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Load from models/market_context.json if available; else synthesize hourly."""
    mpath = models_dir / "market_context.json"
    out: Dict[str, List[Dict[str, Any]]] = {}

    if mpath.exists():
        try:
            data = json.loads(mpath.read_text())
            # Expected shape (used elsewhere): {"series": {"BTC":[{"ts":..,"price":..},...] , ...}}
            series = data.get("series") or data.get("prices") or {}
            # Try common coin name mapping to tickers
            map_keys = {
                "bitcoin": "BTC", "BTC": "BTC",
                "ethereum": "ETH", "ETH": "ETH",
                "solana": "SOL", "SOL": "SOL",
            }
            for k, v in series.items():
                sym = map_keys.get(k, k.upper())
                if sym in symbols and isinstance(v, list) and v:
                    # Filter by lookback and coerce fields
                    items = []
                    cutoff = _now_utc() - timedelta(hours=lookback_h)
                    for row in v:
                        ts = row.get("ts") or row.get("time") or row.get("date")
                        px = row.get("price") or row.get("close") or row.get("value")
                        if ts is None or px is None:
                            continue
                        dt = _to_dt(ts) if "T" in ts else datetime.fromisoformat(ts + "T00:00:00+00:00")
                        if dt >= cutoff:
                            items.append({"ts": _iso(dt), "price": float(px)})
                    items.sort(key=lambda r: r["ts"])
                    if items:
                        out[sym] = items
        except Exception:
            out = {}

    if len(out) == len(symbols):
        return out

    # Synthesize hourly prices (gentle drift) for any missing symbols
    missing = [s for s in symbols if s not in out]
    if missing:
        start = _now_utc() - timedelta(hours=lookback_h + 1)
        hours = lookback_h + 1
        for s in missing:
            px = 100.0 if s == "SOL" else (2000.0 if s == "ETH" else 60_000.0)
            series = []
            for i in range(hours + 1):
                dt = start + timedelta(hours=i)
                # deterministic pseudo-random walk (no numpy)
                drift = (math.sin(i * 0.7 + len(s)) * 0.001)  # ~0.1% wiggle
                px = px * (1.0 + drift)
                series.append({"ts": _iso(dt), "price": float(px)})
            out[s] = series

    return out


# ---------------------------
# Core simulator
# ---------------------------

def run_paper_trader(ctx, mode: str = "backtest") -> Dict[str, Any]:
    """
    Simulate trading using MoonWire signals.
    Returns dict with keys: trades_path, equity_path, metrics_path, artifacts (list).
    """
    # Resolve dirs (support both SummaryContext and direct shims)
    logs_dir: Path = getattr(ctx, "logs_dir", Path("logs"))
    models_dir: Path = getattr(ctx, "models_dir", Path("models"))
    artifacts_dir: Path = getattr(ctx, "artifacts_dir", Path("artifacts"))
    ensure_dir(logs_dir); ensure_dir(models_dir); ensure_dir(artifacts_dir)

    # Env knobs
    symbols = [s.strip().upper() for s in os.getenv("MW_PERF_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()]
    lookback_h = int(os.getenv("MW_PERF_LOOKBACK_H", "72"))
    horizon_min = int(os.getenv("MW_PERF_HORIZON_MIN", "60"))
    slippage = _bps(float(os.getenv("MW_PERF_SLIPPAGE_BPS", "2")))
    fees = _bps(float(os.getenv("MW_PERF_FEES_BPS", "1")))
    capital = float(os.getenv("MW_PERF_CAPITAL", "100000"))

    # Load inputs (with demo fallbacks)
    signals = _load_signals(logs_dir, lookback_h, symbols)
    prices = _load_prices(models_dir, lookback_h, symbols)

    # Prepare logs
    trades_path = logs_dir / "trades.jsonl"
    equity_path = logs_dir / "equity_curve.jsonl"
    trades_path.write_text("")  # truncate
    equity_path.write_text("")

    # Build initial equity point at earliest price ts
    earliest = None
    for s in symbols:
        if s in prices and prices[s]:
            ts = _to_dt(prices[s][0]["ts"])
            earliest = ts if earliest is None else min(earliest, ts)
    if earliest is None:
        earliest = _now_utc()
    equity_curve: List[Dict[str, Any]] = [{"ts": _iso(earliest), "equity": capital}]
    equity_path.write_text(json.dumps(equity_curve[0]) + "\n")

    # Bucket signals by symbol
    by_sym: Dict[str, List[Dict[str, Any]]] = {s: [] for s in symbols}
    for r in signals:
        if r["symbol"] in by_sym:
            by_sym[r["symbol"]].append(r)
    for s in symbols:
        by_sym[s].sort(key=lambda r: r["ts"])

    trades: List[Dict[str, Any]] = []

    # Simulate independent per-symbol PnL with compounding on aggregate equity
    for sym in symbols:
        if not prices.get(sym):
            continue
        sym_prices = prices[sym]
        sym_signals = by_sym.get(sym, [])
        if not sym_signals:
            continue

        pos = 0  # -1, 0, +1
        entry_px = None
        entry_ts = None

        for idx, sig in enumerate(sym_signals):
            sig_dt = _to_dt(sig["ts"])
            dirn = 1 if str(sig.get("direction", "long")).lower().startswith("l") else -1

            # Determine exit time for any open position if reverse signal or time horizon triggers before new entry
            # First handle time-based exits for current position before processing new signal
            if pos != 0 and entry_ts is not None:
                time_exit_dt = entry_ts + timedelta(minutes=horizon_min)
                # If the new signal arrives after our time exit, close at time_exit first
                if time_exit_dt <= sig_dt:
                    exit_dt, raw_exit_px = _nearest_price_after(sym_prices, time_exit_dt)
                    # costs at entry and exit (fees applied both sides)
                    gross = (raw_exit_px / entry_px) - 1.0
                    gross = gross * pos
                    cost = (slippage + fees) * 2.0
                    pnl_frac = gross - cost
                    trade = {
                        "ts": _iso(exit_dt),
                        "symbol": sym,
                        "side": "long" if pos == 1 else "short",
                        "entry_ts": _iso(entry_ts),
                        "entry": entry_px,
                        "exit": raw_exit_px,
                        "pnl_frac": pnl_frac,
                    }
                    trades.append(trade)
                    trades_path.write_text(trades_path.read_text() + json.dumps(trade) + "\n")
                    # Update equity
                    capital = capital * (1.0 + pnl_frac)
                    ept = {"ts": _iso(exit_dt), "equity": capital}
                    equity_curve.append(ept)
                    equity_path.write_text(equity_path.read_text() + json.dumps(ept) + "\n")
                    pos = 0
                    entry_px = None
                    entry_ts = None

            # Process the incoming signal after any time exit
            # If same direction as existing position, ignore; if opposite, close and flip
            if pos == 0:
                # enter new position at next available price after sig time
                entry_dt, raw_entry_px = _nearest_price_after(sym_prices, sig_dt)
                # apply slippage/fees on entry only to effective price
                entry_px = raw_entry_px * (1.0 + slippage * dirn)  # worse price for entry
                entry_ts = entry_dt
                pos = dirn
            else:
                # If signal suggests reverse
                new_dir = dirn
                if new_dir != pos:
                    # close existing at current signal time (first available price)
                    exit_dt, raw_exit_px = _nearest_price_after(sym_prices, sig_dt)
                    gross = (raw_exit_px / entry_px) - 1.0
                    gross = gross * pos
                    cost = (slippage + fees) * 2.0
                    pnl_frac = gross - cost
                    trade = {
                        "ts": _iso(exit_dt),
                        "symbol": sym,
                        "side": "long" if pos == 1 else "short",
                        "entry_ts": _iso(entry_ts),
                        "entry": entry_px,
                        "exit": raw_exit_px,
                        "pnl_frac": pnl_frac,
                    }
                    trades.append(trade)
                    trades_path.write_text(trades_path.read_text() + json.dumps(trade) + "\n")
                    capital = capital * (1.0 + pnl_frac)
                    ept = {"ts": _iso(exit_dt), "equity": capital}
                    equity_curve.append(ept)
                    equity_path.write_text(equity_path.read_text() + json.dumps(ept) + "\n")

                    # flip into new position
                    entry_dt, raw_entry_px = _nearest_price_after(sym_prices, sig_dt)
                    entry_px = raw_entry_px * (1.0 + slippage * new_dir)
                    entry_ts = entry_dt
                    pos = new_dir
                # else same direction: keep running; horizon will handle exit

        # Close any residual position by horizon after last signal
        if pos != 0 and entry_ts is not None:
            exit_dt, raw_exit_px = _nearest_price_after(sym_prices, entry_ts + timedelta(minutes=horizon_min))
            gross = (raw_exit_px / entry_px) - 1.0
            gross = gross * pos
            cost = (slippage + fees) * 2.0
            pnl_frac = gross - cost
            trade = {
                "ts": _iso(exit_dt),
                "symbol": sym,
                "side": "long" if pos == 1 else "short",
                "entry_ts": _iso(entry_ts),
                "entry": entry_px,
                "exit": raw_exit_px,
                "pnl_frac": pnl_frac,
            }
            trades.append(trade)
            trades_path.write_text(trades_path.read_text() + json.dumps(trade) + "\n")
            capital = capital * (1.0 + pnl_frac)
            ept = {"ts": _iso(exit_dt), "equity": capital}
            equity_curve.append(ept)
            equity_path.write_text(equity_path.read_text() + json.dumps(ept) + "\n")

    # ---------------------------
    # Metrics + artifacts
    # ---------------------------
    # Build per-symbol curves and compute metrics
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        t_sym = [t for t in trades if t["symbol"] == sym]
        if not t_sym:
            continue
        eq = float(equity_curve[0]["equity"])
        curve = [{"ts": equity_curve[0]["ts"], "equity": eq}]
        for t in t_sym:
            eq *= (1.0 + float(t["pnl_frac"]))
            curve.append({"ts": t["ts"], "equity": eq})
        by_symbol[sym] = compute_metrics(curve, t_sym)

    aggregate = compute_metrics(equity_curve, trades)
    payload = {
        "generated_at": aggregate["generated_at"],
        "mode": os.getenv("MW_PERF_MODE", "backtest"),
        "window_hours": lookback_h,
        "capital": float(equity_curve[0]["equity"]),
        "by_symbol": by_symbol,
        "aggregate": {k: v for k, v in aggregate.items() if k not in ("generated_at",)},
        "demo": not (logs_dir / "signals.jsonl").exists(),
    }
    metrics_path = models_dir / "performance_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2))

    # --- plots ---
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa

        # Equity curve
        xs = [ _to_dt(p["ts"]) for p in equity_curve ]
        ys = [ float(p["equity"]) for p in equity_curve ]
        fig = plt.figure()
        plt.plot(xs, ys)
        plt.title("Equity Curve")
        plt.xlabel("Time"); plt.ylabel("Equity")
        p1 = artifacts_dir / "perf_equity_curve.png"
        fig.savefig(str(p1)); plt.close(fig)

        # Drawdown curve
        dd = []
        peak = -1e18
        for v in ys:
            peak = max(peak, v)
            dd.append((v - peak) / peak if peak > 0 else 0.0)
        fig = plt.figure()
        plt.plot(xs, dd)
        plt.title("Drawdown")
        plt.xlabel("Time"); plt.ylabel("Drawdown")
        p2 = artifacts_dir / "perf_drawdown.png"
        fig.savefig(str(p2)); plt.close(fig)

        # Returns histogram
        rets = []
        for i in range(1, len(ys)):
            if ys[i-1] != 0:
                rets.append(ys[i] / ys[i-1] - 1.0)
        fig = plt.figure()
        plt.hist(rets, bins=10)
        plt.title("Returns Histogram")
        plt.xlabel("Return"); plt.ylabel("Count")
        p3 = artifacts_dir / "perf_returns_hist.png"
        fig.savefig(str(p3)); plt.close(fig)

        # By-symbol bar (Sharpe if available)
        labels, vals = [], []
        for sym, m in by_symbol.items():
            labels.append(sym); vals.append(float(m.get("sharpe") or 0.0))
        if labels:
            fig = plt.figure()
            plt.bar(labels, vals)
            plt.title("P&L by Symbol (Sharpe)")
            plt.xlabel("Symbol"); plt.ylabel("Sharpe")
            p4 = artifacts_dir / "perf_by_symbol_bar.png"
            fig.savefig(str(p4)); plt.close(fig)
        else:
            p4 = artifacts_dir / "perf_by_symbol_bar.png"  # create empty stub
            if not p4.exists():
                p4.write_bytes(b"")

        artifacts = [str(p1), str(p2), str(p3), str(p4)]
    except Exception:
        artifacts = []

    return {
        "trades_path": str(trades_path),
        "equity_path": str(equity_path),
        "metrics_path": str(metrics_path),
        "artifacts": artifacts,
        "n_trades": len(trades),
    }