# scripts/perf/paper_trader.py
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Matplotlib strictly on Agg; never let GUI backends load in CI
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

# Local metrics helper
try:
    from scripts.perf.performance_metrics import compute_metrics
except Exception:
    # very defensive: tiny fallback
    def compute_metrics(equity_series, returns_series, trades):
        return {
            "sharpe": None, "sortino": None, "max_drawdown": None,
            "calmar": None, "win_rate": None, "profit_factor": None,
            "avg_trade": None, "exposure_pct": None, "cagr": None,
        }


# ------------ helpers

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _fsync_text(path: Path, text: str, mode: str = "w", encoding: str = "utf-8") -> None:
    """Write text robustly and fsync while file descriptor is open."""
    _ensure_dir(path.parent)
    with path.open(mode, encoding=encoding, newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

def _fsync_bytes(path: Path, data: bytes, mode: str = "wb") -> None:
    _ensure_dir(path.parent)
    with path.open(mode) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Ctx:
    # file roots
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    # knobs
    demo_mode: bool = False
    symbols: List[str] = field(default_factory=lambda: [s.strip() for s in os.getenv("MW_PERF_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()])
    horizon_min: int = int(os.getenv("MW_PERF_HORIZON_MIN", "60"))
    slippage_bps: float = float(os.getenv("MW_PERF_SLIPPAGE_BPS", "2"))
    fees_bps: float = float(os.getenv("MW_PERF_FEES_BPS", "1"))
    capital: float = float(os.getenv("MW_PERF_CAPITAL", "100000"))
    risk_free: float = float(os.getenv("MW_PERF_RISK_FREE", "0.0"))
    lookback_h: int = int(os.getenv("MW_PERF_LOOKBACK_H", "72"))

    @property
    def signals_file(self) -> Optional[Path]:
        override = os.getenv("SIGNALS_FILE")
        if override:
            return Path(override)
        # default dual-write legacy/canonical readers: prefer canonical
        primary = self.logs_dir / "signal_history.jsonl"
        legacy = self.logs_dir / "signals.jsonl"
        if primary.exists():
            return primary
        if legacy.exists():
            return legacy
        return primary  # default target if we need to synthesize

# ------------ core IO

def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _read_market_context(models_dir: Path) -> Optional[Dict[str, Any]]:
    j = models_dir / "market_context.json"
    if not j.exists():
        return None
    try:
        return json.loads(j.read_text(encoding="utf-8"))
    except Exception:
        return None

# ------------ demo synthesizers (so CI never goes empty)

def _synth_signals(ctx: Ctx) -> List[Dict[str, Any]]:
    now = _now_utc()
    rows: List[Dict[str, Any]] = []
    for s in ctx.symbols:
        for k in range(3):  # 3 per symbol
            ts = now - timedelta(hours=min(ctx.lookback_h - 1, 3*(k+1)))
            direction = "long" if k % 2 == 0 else "short"
            row = {
                "id": f"sig_{_iso(ts)}_{s}_{direction}",
                "ts": _iso(ts),
                "symbol": s.upper(),
                "direction": direction.lower(),
                "confidence": 0.70 + 0.05 * (k % 2),
                "price": 100.0 + 10.0 * k,  # seed price; sim will use series
                "source": "demo",
                "model_version": "v0.9.0",
                "outcome": None,
            }
            rows.append(row)
    # dual-write only if no override in env (mirror Task 0 policy)
    if not os.getenv("SIGNALS_FILE"):
        _ensure_dir(ctx.logs_dir)
        _fsync_text(ctx.logs_dir / "signal_history.jsonl", "", mode="a")  # ensure file exists
        _fsync_text(ctx.logs_dir / "signals.jsonl", "", mode="a")
        for r in rows:
            _fsync_text(ctx.logs_dir / "signal_history.jsonl", json.dumps(r) + "\n", mode="a")
            _fsync_text(ctx.logs_dir / "signals.jsonl", json.dumps(r) + "\n", mode="a")
    else:
        path = Path(os.getenv("SIGNALS_FILE"))
        _fsync_text(path, "", mode="a")
        for r in rows:
            _fsync_text(path, json.dumps(r) + "\n", mode="a")
    return rows

def _synth_price_series(base: float, n: int, drift: float = 0.0, vol: float = 0.002, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(n)
    series = [base]
    for r in rets:
        series.append(series[-1] * (1.0 + r))
    return np.array(series)

def _build_price_book(ctx: Ctx) -> Dict[str, Dict[str, Any]]:
    """
    Build a simple minute-level price book for each symbol over lookback window.
    If models/market_context.json has latest spot, anchor around that; else use a default.
    """
    lookback_minutes = ctx.lookback_h * 60
    mc = _read_market_context(ctx.models_dir)
    anchors: Dict[str, float] = {}
    if mc and "prices" in mc:
        # expected shape (your existing emitter): {"prices": {"BTC": {"current": 60000.0}, ...}}
        for s in ctx.symbols:
            anchors[s] = float(mc["prices"].get(s, {}).get("current", 100.0))
    else:
        for s in ctx.symbols:
            anchors[s] = 100.0

    book: Dict[str, Dict[str, Any]] = {}
    now = _now_utc()
    times = [now - timedelta(minutes=lookback_minutes - i) for i in range(lookback_minutes + 1)]
    for sym in ctx.symbols:
        arr = _synth_price_series(anchors[sym], lookback_minutes, drift=0.0, vol=0.0015, seed=hash(sym) % 2**32)
        book[sym] = {"times": times, "prices": arr}
    return book

def _price_at(book: Dict[str, Dict[str, Any]], symbol: str, ts: datetime) -> Optional[float]:
    row = book.get(symbol)
    if not row:
        return None
    times: List[datetime] = row["times"]  # type: ignore
    prices: np.ndarray = row["prices"]    # type: ignore
    # find first index >= ts
    # times guaranteed sorted
    lo, hi = 0, len(times) - 1
    if ts <= times[0]:
        return float(prices[0])
    if ts >= times[-1]:
        return float(prices[-1])
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < ts:
            lo = mid + 1
        else:
            hi = mid
    return float(prices[lo])

# ------------ simulation

@dataclass
class Trade:
    ts_entry: datetime
    ts_exit: datetime
    symbol: str
    side: str  # "long" or "short"
    entry: float
    exit: float
    pnl: float
    pnl_pct: float

def _simulate(ctx: Ctx, signals: List[Dict[str, Any]], book: Dict[str, Dict[str, Any]]) -> Tuple[List[Trade], List[Tuple[datetime, float]]]:
    signals = [s for s in signals if s.get("symbol") in ctx.symbols]
    # group by symbol, sort by ts
    by_sym: Dict[str, List[Dict[str, Any]]] = {s: [] for s in ctx.symbols}
    for s in signals:
        try:
            by_sym[s["symbol"]].append(s)
        except Exception:
            pass
    for s in by_sym:
        by_sym[s].sort(key=lambda r: r["ts"])

    equity = ctx.capital
    equity_curve: List[Tuple[datetime, float]] = [( _now_utc() - timedelta(hours=ctx.lookback_h), equity )]
    trades: List[Trade] = []

    slip = ctx.slippage_bps / 10000.0
    fee  = ctx.fees_bps / 10000.0
    horizon = timedelta(minutes=ctx.horizon_min)

    for sym, rows in by_sym.items():
        open_side: Optional[str] = None
        open_entry: Optional[float] = None
        open_ts: Optional[datetime] = None

        for idx, sig in enumerate(rows):
            ts = datetime.fromisoformat(sig["ts"].replace("Z", "+00:00"))
            side = str(sig.get("direction", "long")).lower()
            # entry at next bar: approximate with price at ts (minute-grid)
            price = _price_at(book, sym, ts)
            if price is None:
                continue

            # close if reverse
            if open_side and side != open_side:
                exit_ts = ts
                exit_price = _price_at(book, sym, exit_ts) or price
                signed = 1.0 if open_side == "long" else -1.0
                gross = signed * (exit_price - open_entry) / open_entry
                gross -= (2*slip + 2*fee)
                pnl = equity * gross
                equity += pnl
                trades.append(Trade(open_ts, exit_ts, sym, open_side, open_entry, exit_price, pnl, gross))
                equity_curve.append((exit_ts, equity))
                open_side = None
                open_entry = None
                open_ts = None

            # open (or flip already handled)
            open_side = side
            # pay entry slippage/fee via effective price
            eff = price * (1 + slip + fee) if side == "long" else price * (1 - slip - fee)
            open_entry = eff
            open_ts = ts

            # check horizon exit vs next signal
            exit_deadline = ts + horizon
            # exit at horizon (unless a reverse came earlier which we handled)
            exit_price = _price_at(book, sym, exit_deadline) or eff
            signed = 1.0 if side == "long" else -1.0
            gross = signed * (exit_price - eff) / eff
            gross -= (slip + fee)  # exit costs
            pnl = equity * gross
            equity += pnl
            trades.append(Trade(ts, exit_deadline, sym, side, eff, exit_price, pnl, gross))
            equity_curve.append((exit_deadline, equity))
            open_side = None
            open_entry = None
            open_ts = None

    # sort equity by time (across symbols)
    equity_curve.sort(key=lambda t: t[0])
    return trades, equity_curve

# ------------ public API

def run_paper_trader(ctx: Ctx, mode: str = "backtest") -> Dict[str, Any]:
    """
    Runs a simple time-horizon strategy backtest (or appends in live mode).
    Emits:
      - logs/trades.jsonl
      - logs/equity_curve.jsonl
      - artifacts/perf_equity_curve.png, perf_drawdown.png, perf_returns_hist.png, perf_by_symbol_bar.png
      - models/performance_metrics.json
    Returns the metrics dict.
    """
    _ensure_dir(ctx.logs_dir); _ensure_dir(ctx.models_dir); _ensure_dir(ctx.artifacts_dir)

    # 1) Load/synthesize signals
    sig_path = ctx.signals_file or (ctx.logs_dir / "signal_history.jsonl")
    signals = list(_read_jsonl(sig_path))
    if not signals and ctx.demo_mode:
        signals = _synth_signals(ctx)

    # 2) Build price book
    book = _build_price_book(ctx)

    # 3) Simulate
    trades, equity_curve = _simulate(ctx, signals, book)

    # 4) Write logs (fsync-safe)
    trades_path = ctx.logs_dir / "trades.jsonl"
    equity_path = ctx.logs_dir / "equity_curve.jsonl"
    # rewrite fresh each run (simpler in CI)
    _fsync_text(trades_path, "", mode="w")
    for t in trades:
        _fsync_text(trades_path, json.dumps({
            "ts_entry": _iso(t.ts_entry), "ts_exit": _iso(t.ts_exit),
            "symbol": t.symbol, "side": t.side, "qty": 1.0,
            "entry": t.entry, "exit": t.exit, "pnl": t.pnl, "pnl_pct": t.pnl_pct
        }) + "\n", mode="a")
    _fsync_text(equity_path, "", mode="w")
    for ts, eq in equity_curve:
        _fsync_text(equity_path, json.dumps({"ts": _iso(ts), "equity": eq}) + "\n", mode="a")

    # 5) Compute metrics
    eq_vals = np.array([eq for _, eq in equity_curve], dtype=float)
    rets = np.diff(eq_vals) / (eq_vals[:-1] + 1e-12) if len(eq_vals) > 1 else np.array([])
    metrics = compute_metrics(eq_vals, rets, trades)

    # 6) Write metrics JSON
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for s in ctx.symbols:
        sym_trades = [t for t in trades if t.symbol == s]
        eq_s = ctx.capital
        eq_series_s = [eq_s]
        for t in sym_trades:
            eq_s += t.pnl
            eq_series_s.append(eq_s)
        eq_series_s = np.array(eq_series_s, dtype=float)
        rets_s = np.diff(eq_series_s) / (eq_series_s[:-1] + 1e-12) if len(eq_series_s) > 1 else np.array([])
        m_s = compute_metrics(eq_series_s, rets_s, sym_trades)
        by_symbol[s] = {
            "trades": len(sym_trades),
            "sharpe": m_s["sharpe"],
            "sortino": m_s["sortino"],
            "max_drawdown": m_s["max_drawdown"],
            "win_rate": m_s["win_rate"],
            "profit_factor": m_s["profit_factor"],
            "cagr": m_s.get("cagr", None),
        }

    agg = {
        "trades": len(trades),
        "sharpe": metrics["sharpe"],
        "sortino": metrics["sortino"],
        "max_drawdown": metrics["max_drawdown"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "cagr": metrics.get("cagr", None),
    }

    out = {
        "generated_at": _iso(_now_utc()),
        "mode": mode,
        "window_hours": ctx.lookback_h,
        "capital": ctx.capital,
        "by_symbol": by_symbol,
        "aggregate": agg,
        "demo": bool(ctx.demo_mode),
    }
    _fsync_text(ctx.models_dir / "performance_metrics.json", json.dumps(out, indent=2) + "\n", mode="w")

    # 7) Plots (no fsync needed; matplotlib handles file)
    if len(equity_curve) > 0:
        # equity curve
        ts = [t for t, _ in equity_curve]
        eq = [e for _, e in equity_curve]
        plt.figure()
        plt.plot(ts, eq)
        plt.title("Equity Curve")
        plt.xlabel("Time"); plt.ylabel("Equity")
        plt.tight_layout()
        plt.savefig(ctx.artifacts_dir / "perf_equity_curve.png")
        plt.close()

        # drawdown
        arr = np.array(eq, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / (peak + 1e-12)
        plt.figure()
        plt.plot(ts, dd)
        plt.title("Drawdown")
        plt.xlabel("Time"); plt.ylabel("Drawdown")
        plt.tight_layout()
        plt.savefig(ctx.artifacts_dir / "perf_drawdown.png")
        plt.close()

    # returns hist
    if rets.size > 0:
        plt.figure()
        plt.hist(rets, bins=20)
        plt.title("Per-Step Returns Histogram")
        plt.xlabel("Return"); plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(ctx.artifacts_dir / "perf_returns_hist.png")
        plt.close()

    # by-symbol bar
    labels = list(by_symbol.keys())
    if labels:
        vals = [by_symbol[s]["profit_factor"] if by_symbol[s]["profit_factor"] is not None else 0.0 for s in labels]
        plt.figure()
        plt.bar(labels, vals)
        plt.title("Profit Factor by Symbol")
        plt.tight_layout()
        plt.savefig(ctx.artifacts_dir / "perf_by_symbol_bar.png")
        plt.close()

    return out
