import json
import os
import math
import numpy as np
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from scripts.perf.performance_metrics import compute_metrics

ART_DIR = Path("artifacts")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")

def _now_utc() -> dt.datetime:
    return dt.datetime.utcnow().replace(microsecond=0)

def _iso(ts: dt.datetime) -> str:
    return ts.replace(microsecond=0).isoformat() + "Z"

def _load_signals() -> List[Dict]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    primary = LOGS_DIR / "signal_history.jsonl"
    legacy = LOGS_DIR / "signals.jsonl"
    for p in [primary, legacy]:
      if p.exists() and p.stat().st_size > 0:
        rows = []
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
            return rows
    return []

def _synth_signals() -> List[Dict]:
    base = _now_utc() - dt.timedelta(hours=3)
    syms = ["BTC","ETH","SOL"]
    dirs = ["long","short"]
    rows = []
    for i in range(9):
        ts = base + dt.timedelta(minutes=20*i)
        sym = syms[i % len(syms)]
        direction = dirs[i % 2]
        price = 100.0 + i  # arbitrary
        rows.append({
            "id": f"sig_{_iso(ts)}_{sym}_{direction}",
            "ts": _iso(ts),
            "symbol": sym,
            "direction": direction,
            "confidence": 0.70 + (i%3)*0.05,
            "price": price,
            "source": "demo",
            "model_version": "v0.9.0",
            "outcome": None
        })
    return rows

def _price_series_from_signals(signals: List[Dict], minutes: int = 240) -> List[Tuple[str,float]]:
    # deterministic pseudo-random walk seeded by count
    rng = np.random.default_rng(42)
    if not signals:
        t0 = _now_utc() - dt.timedelta(minutes=minutes)
        p0 = 100.0
    else:
        t0 = dt.datetime.fromisoformat(signals[0]["ts"].replace("Z","+00:00")).replace(tzinfo=None)
        p0 = float(signals[0].get("price", 100.0))
    prices = []
    p = p0
    ts = t0
    for _ in range(minutes):
        step = rng.normal(0.0, 0.1)
        p = max(1.0, p * (1.0 + step/100.0))
        prices.append((_iso(ts), float(p)))
        ts += dt.timedelta(minutes=1)
    return prices

def _price_at(prices: List[Tuple[str,float]], when: dt.datetime) -> Optional[float]:
    # find first price at or after 'when'
    for ts, px in prices:
        t = dt.datetime.fromisoformat(ts.replace("Z","+00:00")).replace(tzinfo=None)
        if t >= when:
            return px
    return prices[-1][1] if prices else None

def _simulate(signals: List[Dict], prices: List[Tuple[str,float]], horizon_min: int, slippage_bps: float, fees_bps: float, capital: float):
    # very simple 1x notional per trade (qty derived so notional = capital)
    trades = []
    equity = []
    eq = capital
    equity.append((prices[0][0], eq))
    open_pos = None  # dict with entry info

    for sig in sorted(signals, key=lambda r: r["ts"]):
        s_ts = dt.datetime.fromisoformat(sig["ts"].replace("Z","+00:00")).replace(tzinfo=None)
        entry_px = _price_at(prices, s_ts)
        if entry_px is None:  # no price
            continue
        side = 1 if sig["direction"].lower() == "long" else -1
        # close existing if opposite
        if open_pos and open_pos["side"] != side:
            # close at signal time (reverse)
            exit_px = entry_px
            pnl_pct = (exit_px - open_pos["entry_px"]) / open_pos["entry_px"] * open_pos["side"]
            slippage = (slippage_bps + fees_bps) / 1e4
            pnl_pct -= slippage
            pnl = eq * pnl_pct
            eq += pnl
            trades.append({
                "symbol": open_pos["symbol"],
                "side": "long" if open_pos["side"] == 1 else "short",
                "entry_ts": _iso(open_pos["entry_ts"]),
                "exit_ts": _iso(s_ts),
                "entry_px": open_pos["entry_px"],
                "exit_px": exit_px,
                "pnl": float(pnl),
                "pnl_pct": float(pnl_pct),
                "closed": True
            })
            open_pos = None
            equity.append((_iso(s_ts), eq))

        # open new
        open_pos = {
            "symbol": sig["symbol"],
            "side": side,
            "entry_ts": s_ts,
            "entry_px": entry_px
        }

        # time based exit
        exit_time = s_ts + dt.timedelta(minutes=horizon_min)
        exit_px = _price_at(prices, exit_time)
        if exit_px is None:
            exit_px = prices[-1][1]
        pnl_pct = (exit_px - open_pos["entry_px"]) / open_pos["entry_px"] * open_pos["side"]
        slippage = (slippage_bps + fees_bps) / 1e4
        pnl_pct -= slippage
        pnl = eq * pnl_pct
        eq += pnl
        trades.append({
            "symbol": open_pos["symbol"],
            "side": "long" if open_pos["side"] == 1 else "short",
            "entry_ts": _iso(open_pos["entry_ts"]),
            "exit_ts": _iso(exit_time),
            "entry_px": open_pos["entry_px"],
            "exit_px": exit_px,
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct),
            "closed": True
        })
        open_pos = None
        equity.append((_iso(exit_time), eq))

    # ensure strictly increasing times in equity
    equity = sorted(list({t: v for t, v in equity}.items()))
    equity = [(t, v) for t, v in equity]
    # compute returns from equity
    rets = []
    for i in range(1, len(equity)):
        prev = equity[i-1][1]
        curr = equity[i][1]
        rets.append((curr - prev) / prev if prev > 0 else 0.0)
    return trades, equity, np.array(rets, dtype=float)

def _plot_equity_and_drawdown(equity: List[tuple]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not equity:
        return
    ts = [e[0] for e in equity]
    vals = [e[1] for e in equity]

    # Equity curve
    plt.figure()
    plt.plot(ts, vals)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    (ART_DIR / "perf_equity_curve.png").write_bytes(plt.gcf().canvas.print_png(bytesio := bytearray()) or bytes(bytesio))
    plt.close()

    # Drawdown
    arr = np.array(vals, dtype=float)
    peaks = np.maximum.accumulate(arr)
    dd = (arr - peaks) / np.where(peaks == 0, 1.0, peaks)
    plt.figure()
    plt.plot(ts, dd)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    (ART_DIR / "perf_drawdown.png").write_bytes(plt.gcf().canvas.print_png(bytesio := bytearray()) or bytes(bytesio))
    plt.close()

def _plot_returns_hist(rets: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure()
    plt.hist(rets, bins=20)
    plt.tight_layout()
    (ART_DIR / "perf_returns_hist.png").write_bytes(plt.gcf().canvas.print_png(bytesio := bytearray()) or bytes(bytesio))
    plt.close()

def _plot_by_symbol(trades: List[Dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {}
    for t in trades:
        s = t["symbol"]
        by.setdefault(s, []).append(t["pnl"])
    labels = sorted(by.keys())
    totals = [sum(by[k]) for k in labels]
    plt.figure()
    plt.bar(labels, totals)
    plt.tight_layout()
    (ART_DIR / "perf_by_symbol_bar.png").write_bytes(plt.gcf().canvas.print_png(bytesio := bytearray()) or bytes(bytesio))
    plt.close()

def run_paper_trader(ctx, mode: str = "backtest") -> Dict:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    demo = os.getenv("DEMO_MODE", "true").lower() == "true"
    symbols = [s.strip().upper() for s in os.getenv("MW_PERF_SYMBOLS","BTC,ETH,SOL").split(",") if s.strip()]
    horizon_min = int(os.getenv("MW_PERF_HORIZON_MIN","60"))
    slippage_bps = float(os.getenv("MW_PERF_SLIPPAGE_BPS","2"))
    fees_bps = float(os.getenv("MW_PERF_FEES_BPS","1"))
    capital = float(os.getenv("MW_PERF_CAPITAL","100000"))
    risk_free = float(os.getenv("MW_PERF_RISK_FREE","0.0"))
    lookback_h = int(os.getenv("MW_PERF_LOOKBACK_H","72"))

    sigs = _load_signals()
    if not sigs and demo:
        sigs = _synth_signals()

    # filter to symbols of interest
    if symbols:
        sigs = [s for s in sigs if s.get("symbol","").upper() in symbols]

    # build synthetic price series around signals (demo-safe)
    prices = _price_series_from_signals(sigs, minutes=lookback_h*60 if lookback_h>0 else 240)

    trades, equity, rets = _simulate(sigs, prices, horizon_min, slippage_bps, fees_bps, capital)
    metrics = compute_metrics(equity, rets, trades, risk_free=risk_free)

    # persist logs
    (LOGS_DIR / "trades.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in trades),
        encoding="utf-8"
    )
    (LOGS_DIR / "equity_curve.jsonl").write_text(
        "".join(json.dumps({"ts": t, "equity": v}) + "\n" for t, v in equity),
        encoding="utf-8"
    )

    # plots
    _plot_equity_and_drawdown(equity)
    if rets.size > 0:
        _plot_returns_hist(rets)
    _plot_by_symbol(trades)

    out = {
        "generated_at": _iso(_now_utc()),
        "mode": mode,
        "window_hours": lookback_h,
        "capital": capital,
        "by_symbol": {},
        "aggregate": metrics
    }
    # per-symbol rollups (simple)
    bysym = {}
    for t in trades:
        s = t["symbol"]
        bysym.setdefault(s, []).append(t)
    for s, ts in bysym.items():
        rets_s = np.array([tt["pnl_pct"] for tt in ts], dtype=float)
        eq_s = []
        eq = capital
        now = _now_utc()
        for i, r in enumerate(rets_s):
            eq *= (1.0 + r)
            eq_s.append((_iso(now + dt.timedelta(minutes=i)), eq))
        out["by_symbol"][s] = compute_metrics(eq_s, rets_s, ts)

    # write metrics JSON
    (MODELS_DIR / "performance_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
