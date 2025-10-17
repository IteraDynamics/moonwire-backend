#!/usr/bin/env python3
import os
import json
import math
import random
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART_DIR = Path("artifacts")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

@dataclass
class Ctx:
    demo_mode: bool = True

def _utcnow_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _ensure_dirs():
    ART_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

def _load_signals() -> List[dict]:
    # Prefer canonical, then legacy; synthesize if needed
    paths = []
    override = os.getenv("SIGNALS_FILE")
    if override:
        paths.append(Path(override))
    paths += [LOGS_DIR / "signal_history.jsonl", LOGS_DIR / "signals.jsonl"]

    rows: List[dict] = []
    for p in paths:
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
            if rows:
                break
    return rows

def _load_prices_from_market_context(symbols: List[str]) -> Dict[str, List[Tuple[dt.datetime, float]]]:
    out: Dict[str, List[Tuple[dt.datetime, float]]] = {s: [] for s in symbols}
    p = MODELS_DIR / "market_context.json"
    if not p.exists():
        return out
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return out
    # This file in your CI is summary-like; we’ll synthesize a tiny hourly series around the spot price
    coins = {c["symbol"]: float(c["price"]) for c in obj.get("coins", []) if "symbol" in c and "price" in c}
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    hours = int(os.getenv("MW_PERF_LOOKBACK_H", "72"))
    for s in symbols:
        spot = coins.get(s, None)
        if spot is None:
            continue
        series = []
        for h in range(hours, -1, -1):
            ts = now - dt.timedelta(hours=h)
            # make a gentle random walk around spot
            price = spot * (1.0 + 0.005 * math.sin(h/5.0)) * (1.0 + 0.002 * (random.random() - 0.5))
            series.append((ts, float(price)))
        out[s] = series
    return out

def _synthesize_signals(symbols: List[str], n_per_symbol: int = 3) -> List[dict]:
    rows: List[dict] = []
    now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    for s in symbols:
        for i in range(n_per_symbol):
            ts = (now - dt.timedelta(hours=2*i+1)).isoformat() + "Z"
            direction = "long" if i % 2 == 0 else "short"
            rows.append({
                "id": f"sig_{ts}_{s}_{direction}",
                "ts": ts,
                "symbol": s,
                "direction": direction,
                "confidence": 0.7,
                "price": 100.0,
                "source": "demo",
                "model_version": "v0.9.0",
                "outcome": None
            })
    return rows

def _to_dt(ts_iso: str) -> dt.datetime:
    if ts_iso.endswith("Z"):
        ts_iso = ts_iso[:-1]
    return dt.datetime.fromisoformat(ts_iso)

def _price_at_or_after(series: List[Tuple[dt.datetime, float]], t: dt.datetime) -> Optional[float]:
    # assume sorted ascending by time
    for ts, px in series:
        if ts >= t:
            return px
    return series[-1][1] if series else None

def _price_at_or_before(series: List[Tuple[dt.datetime, float]], t: dt.datetime) -> Optional[float]:
    last = None
    for ts, px in series:
        if ts <= t:
            last = px
        else:
            break
    return last

def run_paper_trader(ctx: Ctx, mode: str = "backtest") -> dict:
    """
    Simulate a super-simple strategy: enter 1x notional per signal at next bar,
    exit after HORIZON minutes unless a reverse signal appears earlier.
    Writes:
      - logs/trades.jsonl
      - logs/equity_curve.jsonl
      - artifacts/perf_equity_curve.png, perf_drawdown.png, perf_returns_hist.png, perf_by_symbol_bar.png
      - models/performance_metrics.json
    """
    _ensure_dirs()

    symbols = [s.strip().upper() for s in os.getenv("MW_PERF_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()]
    horizon_min = int(os.getenv("MW_PERF_HORIZON_MIN", "60"))
    slippage_bps = float(os.getenv("MW_PERF_SLIPPAGE_BPS", "2"))
    fees_bps = float(os.getenv("MW_PERF_FEES_BPS", "1"))
    capital = float(os.getenv("MW_PERF_CAPITAL", "100000"))
    window_h = int(os.getenv("MW_PERF_LOOKBACK_H", "72"))

    signals = _load_signals()
    if not signals and ctx.demo_mode:
        signals = _synthesize_signals(symbols, n_per_symbol=3)

    # bucket by symbol and sort
    by_sym: Dict[str, List[dict]] = {s: [] for s in symbols}
    for r in signals:
        sym = str(r.get("symbol", "")).upper()
        if sym in by_sym:
            by_sym[sym].append(r)
    for s in by_sym:
        by_sym[s].sort(key=lambda r: _to_dt(r["ts"]))

    prices = _load_prices_from_market_context(symbols)
    if not any(prices.values()) and ctx.demo_mode:
        # synth price series if missing
        for s in symbols:
            base = 100.0 + 5.0 * symbols.index(s)
            now = dt.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            series = []
            for h in range(window_h, -1, -1):
                ts = now - dt.timedelta(hours=h)
                px = base * (1.0 + 0.01 * math.sin(h/7.0))
                series.append((ts, float(px)))
            prices[s] = series

    # simulate
    equity = capital
    equity_curve: List[Tuple[dt.datetime, float]] = []
    returns: List[float] = []
    trades: List[dict] = []

    s_bps = slippage_bps / 1e4
    f_bps = fees_bps / 1e4

    start_t = dt.datetime.utcnow() - dt.timedelta(hours=window_h)
    for s in symbols:
        sigs = by_sym.get(s, [])
        series = prices.get(s, [])
        if not series:
            continue

        position = 0   # +1 long, -1 short, 0 flat
        entry_px = None
        entry_t = None

        i = 0
        while i < len(sigs):
            sig = sigs[i]
            t_sig = _to_dt(sig["ts"])
            if t_sig < start_t:
                i += 1
                continue

            dirn = str(sig.get("direction", "long")).lower()
            side = 1 if dirn == "long" else -1

            # entry at next available price >= t_sig
            px_entry = _price_at_or_after(series, t_sig)
            if px_entry is None:
                i += 1
                continue
            px_entry = px_entry * (1 + s_bps*side)  # simple slippage

            # If we have an open opposite position, close it first
            if position != 0 and side != position and entry_px is not None and entry_t is not None:
                # exit at current bar (flip)
                px_exit = px_entry  # same timestamp, pay fees once here as exit
                pnl = (px_exit - entry_px) * position
                pnl_after = pnl - f_bps * abs(px_exit + entry_px) / 2.0
                equity += pnl_after
                trades.append({
                    "ts": _utcnow_iso(),
                    "symbol": s,
                    "side": "long" if position == 1 else "short",
                    "qty": 1.0,
                    "entry": float(entry_px),
                    "exit": float(px_exit),
                    "pnl": float(pnl_after),
                    "pnl_pct": float(pnl_after / entry_px),
                })
                position = 0
                entry_px = None
                entry_t = None

            # open / flip to new side
            position = side
            entry_px = px_entry
            entry_t = t_sig

            # determine time-based exit or until next reverse signal
            exit_deadline = t_sig + dt.timedelta(minutes=horizon_min)
            j = i + 1
            reverse_seen = False
            while j < len(sigs):
                sig2 = sigs[j]
                t2 = _to_dt(sig2["ts"])
                if t2 > exit_deadline:
                    break
                d2 = str(sig2.get("direction", "long")).lower()
                if (d2 == "long" and position == -1) or (d2 == "short" and position == 1):
                    # exit at price at t2 (reverse)
                    px_exit = _price_at_or_after(series, t2)
                    if px_exit is None:
                        j += 1
                        continue
                    px_exit = px_exit * (1 - s_bps*position)  # slippage on exit
                    pnl = (px_exit - entry_px) * position
                    pnl_after = pnl - f_bps * abs(px_exit + entry_px) / 2.0
                    equity += pnl_after
                    trades.append({
                        "ts": _utcnow_iso(),
                        "symbol": s,
                        "side": "long" if position == 1 else "short",
                        "qty": 1.0,
                        "entry": float(entry_px),
                        "exit": float(px_exit),
                        "pnl": float(pnl_after),
                        "pnl_pct": float(pnl_after / entry_px),
                    })
                    position = 0
                    entry_px = None
                    entry_t = None
                    reverse_seen = True
                    i = j  # continue from the reverse signal
                    break
                j += 1

            if not reverse_seen and entry_px is not None:
                # time-based exit at deadline
                px_exit = _price_at_or_after(series, exit_deadline)
                if px_exit is None:
                    px_exit = series[-1][1]
                px_exit = px_exit * (1 - s_bps*position)
                pnl = (px_exit - entry_px) * position
                pnl_after = pnl - f_bps * abs(px_exit + entry_px) / 2.0
                equity += pnl_after
                trades.append({
                    "ts": _utcnow_iso(),
                    "symbol": s,
                    "side": "long" if position == 1 else "short",
                    "qty": 1.0,
                    "entry": float(entry_px),
                    "exit": float(px_exit),
                    "pnl": float(pnl_after),
                    "pnl_pct": float(pnl_after / entry_px),
                })
                position = 0
                entry_px = None
                entry_t = None

            # update equity curve at exit points
            equity_curve.append((dt.datetime.utcnow(), equity))
            if len(equity_curve) >= 2:
                r = (equity_curve[-1][1] - equity_curve[-2][1]) / max(equity_curve[-2][1], 1e-9)
                returns.append(float(r))

            i += 1

    # Persist logs
    trades_path = LOGS_DIR / "trades.jsonl"
    eq_path = LOGS_DIR / "equity_curve.jsonl"
    with trades_path.open("a", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")
            f.flush()
    os.fsync(trades_path.open("a").fileno()) if trades_path.exists() else None

    with eq_path.open("a", encoding="utf-8") as f:
        for ts, eq in equity_curve:
            f.write(json.dumps({"ts": ts.replace(microsecond=0).isoformat() + "Z", "equity": float(eq)}) + "\n")
            f.flush()
    os.fsync(eq_path.open("a").fileno()) if eq_path.exists() else None

    # Charts — write straight to files (no in-memory bytearray!)
    if equity_curve:
        # equity
        xs = [x[0] for x in equity_curve]
        ys = [x[1] for x in equity_curve]
        plt.figure()
        plt.plot(xs, ys)
        plt.title("Equity Curve")
        plt.tight_layout()
        plt.savefig(ART_DIR / "perf_equity_curve.png")
        plt.close()

        # drawdown
        cummax = np.maximum.accumulate(ys)
        dd = (np.array(ys) - cummax) / np.maximum(cummax, 1e-9)
        plt.figure()
        plt.plot(xs, dd)
        plt.title("Drawdown")
        plt.tight_layout()
        plt.savefig(ART_DIR / "perf_drawdown.png")
        plt.close()

    # returns hist
    if returns:
        plt.figure()
        plt.hist(returns, bins=20)
        plt.title("Returns Histogram")
        plt.tight_layout()
        plt.savefig(ART_DIR / "perf_returns_hist.png")
        plt.close()

    # by-symbol bar (wins)
    if trades:
        buckets: Dict[str, Tuple[int,int]] = {}
        for t in trades:
            s = t["symbol"]
            win = 1 if t["pnl"] > 0 else 0
            tot, w = buckets.get(s, (0,0))
            buckets[s] = (tot+1, w+win)
        labels = list(buckets.keys())
        wrates = [buckets[k][1] / max(buckets[k][0],1) for k in labels]
        plt.figure()
        plt.bar(labels, wrates)
        plt.title("Win Rate by Symbol")
        plt.tight_layout()
        plt.savefig(ART_DIR / "perf_by_symbol_bar.png")
        plt.close()

    # Simple metrics here (Sharpe/Sortino approximations)
    def sharpe(rets: List[float], rf: float=0.0) -> float:
        if not rets: return float("nan")
        arr = np.array(rets) - rf
        sd = arr.std(ddof=1)
        return float(arr.mean() / sd) if sd > 0 else float("nan")

    def sortino(rets: List[float], rf: float=0.0) -> float:
        if not rets: return float("nan")
        arr = np.array(rets) - rf
        downside = arr[arr < 0]
        ds = downside.std(ddof=1) if len(downside) else 0.0
        return float(arr.mean() / ds) if ds > 0 else float("nan")

    wins = sum(1 for t in trades if t["pnl"] > 0)
    pf = (sum(t["pnl"] for t in trades if t["pnl"] > 0) /
          max(abs(sum(t["pnl"] for t in trades if t["pnl"] < 0)), 1e-9)) if trades else float("nan")
    maxdd = float(min(((y - max(ys[:i+1])) / max(max(ys[:i+1]),1e-9)) for i, y in enumerate(ys))) if equity_curve else 0.0

    metrics = {
        "generated_at": _utcnow_iso(),
        "mode": mode,
        "window_hours": window_h,
        "capital": capital,
        "aggregate": {
            "trades": len(trades),
            "sharpe": sharpe(returns),
            "sortino": sortino(returns),
            "max_drawdown": maxdd,
            "win_rate": (wins / len(trades)) if trades else float("nan"),
            "profit_factor": pf,
        }
    }

    out_json = MODELS_DIR / "performance_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
