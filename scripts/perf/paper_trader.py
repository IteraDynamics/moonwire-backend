# scripts/perf/paper_trader.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

try:
    from scripts.perf.performance_metrics import compute_metrics
except Exception:
    def compute_metrics(equity_series, returns_series, trades):
        return {
            "sharpe": None, "sortino": None, "max_drawdown": None,
            "calmar": None, "win_rate": None, "profit_factor": None,
            "avg_trade": None, "exposure_pct": None, "cagr": None,
        }

# --------------------- helpers

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _fsync_text(path: Path, text: str, mode: str = "w", encoding: str = "utf-8") -> None:
    _ensure_dir(path.parent)
    with path.open(mode, encoding=encoding, newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Ctx:
    logs_dir: Path = field(default_factory=lambda: Path("logs"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))

    demo_mode: bool = field(default_factory=lambda: str(os.getenv("DEMO_MODE", os.getenv("MW_DEMO", "false"))).lower() == "true")
    force_demo: bool = field(default_factory=lambda: str(os.getenv("MW_PERF_FORCE_DEMO", "false")).lower() == "true")

    symbols: List[str] = field(default_factory=lambda: [s.strip().upper() for s in os.getenv("MW_PERF_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()])
    horizon_min: int = int(os.getenv("MW_PERF_HORIZON_MIN", "60"))
    slippage_bps: float = float(os.getenv("MW_PERF_SLIPPAGE_BPS", "2"))
    fees_bps: float = float(os.getenv("MW_PERF_FEES_BPS", "1"))
    capital: float = float(os.getenv("MW_PERF_CAPITAL", "100000"))
    risk_free: float = float(os.getenv("MW_PERF_RISK_FREE", "0.0"))
    lookback_h: int = int(os.getenv("MW_PERF_LOOKBACK_H", "72"))
    position_size_pct: float = float(os.getenv("MW_PERF_POSITION_SIZE_PCT", "1.0"))
    stop_loss_pct: Optional[float] = field(default_factory=lambda: float(os.getenv("MW_PERF_STOP_LOSS_PCT")) if os.getenv("MW_PERF_STOP_LOSS_PCT") else None)

    @property
    def signals_file(self) -> Optional[Path]:
        override = os.getenv("SIGNALS_FILE")
        if override:
            return Path(override)
        primary = self.logs_dir / "signal_history.jsonl"
        legacy = self.logs_dir / "signals.jsonl"
        if primary.exists():
            return primary
        if legacy.exists():
            return legacy
        return primary

# --------------------- IO

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

# --------------------- demo synthesizers

def _synth_signals(ctx: Ctx, need_per_symbol: int = 2) -> List[Dict[str, Any]]:
    """Generate at least `need_per_symbol` signals per symbol within lookback window."""
    now = _now_utc()
    rows: List[Dict[str, Any]] = []
    gap_h = max(1, min(6, ctx.lookback_h // (need_per_symbol + 1)))
    for s in ctx.symbols:
        for k in range(need_per_symbol):
            ts = now - timedelta(hours=(k + 1) * gap_h)
            direction = "long" if k % 2 == 0 else "short"
            rows.append({
                "id": f"sig_{_iso(ts)}_{s}_{direction}",
                "ts": _iso(ts),
                "symbol": s.upper(),
                "direction": direction,
                "confidence": 0.70 + 0.05 * (k % 2),
                "price": 100.0 + 5.0 * k,
                "source": "demo",
                "model_version": "v0.9.0",
                "outcome": None,
            })
    return rows

def _synth_price_series(base: float, n: int, drift: float = 0.0, vol: float = 0.0018, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = drift + vol * rng.standard_normal(n)
    series = [base]
    for r in rets:
        series.append(series[-1] * (1.0 + r))
    return np.array(series, dtype=float)

def _build_price_book(ctx: Ctx) -> Dict[str, Dict[str, Any]]:
    lookback_minutes = max(30, ctx.lookback_h * 60)
    mc = _read_market_context(ctx.models_dir)
    anchors: Dict[str, float] = {}
    if mc and "prices" in mc:
        for s in ctx.symbols:
            anchors[s] = float(mc["prices"].get(s, {}).get("current", 100.0))
    else:
        for s in ctx.symbols:
            anchors[s] = 100.0

    book: Dict[str, Dict[str, Any]] = {}
    now = _now_utc()
    times = [now - timedelta(minutes=lookback_minutes - i) for i in range(lookback_minutes + 1)]
    for sym in ctx.symbols:
        arr = _synth_price_series(anchors[sym], lookback_minutes, drift=0.0, vol=0.0018, seed=(hash(sym) & 0xFFFFFFFF))
        book[sym] = {"times": times, "prices": arr}
    return book

def _price_at(book: Dict[str, Dict[str, Any]], symbol: str, ts: datetime) -> Optional[float]:
    row = book.get(symbol)
    if not row:
        return None
    times: List[datetime] = row["times"]  # type: ignore
    prices: np.ndarray = row["prices"]    # type: ignore
    if ts <= times[0]:
        return float(prices[0])
    if ts >= times[-1]:
        return float(prices[-1])
    lo, hi = 0, len(times) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < ts:
            lo = mid + 1
        else:
            hi = mid
    return float(prices[lo])

# --------------------- simulation

from dataclasses import dataclass as _dcd

@_dcd
class Trade:
    ts_entry: datetime
    ts_exit: datetime
    symbol: str
    side: str
    entry: float
    exit: float
    pnl: float
    pnl_pct: float

def _simulate(ctx: Ctx, signals: List[Dict[str, Any]], book: Dict[str, Dict[str, Any]]) -> Tuple[List[Trade], List[Tuple[datetime, float]]]:
    # keep only window signals
    win_start = _now_utc() - timedelta(hours=ctx.lookback_h)
    usable = []
    for s in signals:
        try:
            t = datetime.fromisoformat(str(s["ts"]).replace("Z", "+00:00"))
            if t >= win_start and str(s.get("symbol", "")).upper() in ctx.symbols:
                s2 = dict(s)
                s2["symbol"] = str(s2["symbol"]).upper()
                s2["direction"] = str(s2.get("direction", "long")).lower()
                usable.append(s2)
        except Exception:
            continue

    # group and sort
    by_sym: Dict[str, List[Dict[str, Any]]] = {s: [] for s in ctx.symbols}
    for s in usable:
        by_sym[s["symbol"]].append(s)
    for s in by_sym:
        by_sym[s].sort(key=lambda r: r["ts"])

    equity = float(ctx.capital)
    equity_curve: List[Tuple[datetime, float]] = [(win_start, equity)]
    trades: List[Trade] = []

    slip = ctx.slippage_bps / 10000.0
    fee  = ctx.fees_bps / 10000.0
    horizon = timedelta(minutes=max(15, ctx.horizon_min))

    for sym, rows in by_sym.items():
        for sig in rows:
            ts = datetime.fromisoformat(sig["ts"].replace("Z", "+00:00"))
            side = sig["direction"]
            price = _price_at(book, sym, ts)
            if price is None:
                continue

            eff_entry = price * (1 + slip + fee) if side == "long" else price * (1 - slip - fee)
            exit_ts = ts + horizon
            exit_price = _price_at(book, sym, exit_ts) or eff_entry

            # Check for stop loss during the trade period
            stopped_out = False
            if ctx.stop_loss_pct is not None and ctx.stop_loss_pct > 0:
                # Sample 10 price points during the trade period to check for stop loss
                samples = 10
                for i in range(1, samples + 1):
                    check_ts = ts + timedelta(minutes=(horizon.total_seconds() / 60) * (i / samples))
                    check_price = _price_at(book, sym, check_ts)
                    if check_price is None:
                        continue

                    # Calculate current P&L percentage from entry
                    if side == "long":
                        current_pnl_pct = (check_price - eff_entry) / max(eff_entry, 1e-12)
                        if current_pnl_pct <= -ctx.stop_loss_pct:
                            # Stop loss hit - exit at this price
                            exit_ts = check_ts
                            exit_price = check_price
                            stopped_out = True
                            break
                    else:  # short
                        current_pnl_pct = (eff_entry - check_price) / max(eff_entry, 1e-12)
                        if current_pnl_pct <= -ctx.stop_loss_pct:
                            # Stop loss hit - exit at this price
                            exit_ts = check_ts
                            exit_price = check_price
                            stopped_out = True
                            break

            signed = 1.0 if side == "long" else -1.0
            gross = signed * (exit_price - eff_entry) / max(eff_entry, 1e-12)
            gross -= (slip + fee)  # exit costs

            # Apply position sizing
            position_equity = equity * ctx.position_size_pct
            pnl = position_equity * gross
            equity += pnl

            trades.append(Trade(ts, exit_ts, sym, side, float(eff_entry), float(exit_price), float(pnl), float(gross)))
            equity_curve.append((exit_ts, float(equity)))

    equity_curve.sort(key=lambda t: t[0])
    return trades, equity_curve

# --------------------- main entry

def run_paper_trader(ctx: Ctx, mode: str = "backtest") -> Dict[str, Any]:
    _ensure_dir(ctx.logs_dir); _ensure_dir(ctx.models_dir); _ensure_dir(ctx.artifacts_dir)

    sig_path = ctx.signals_file or (ctx.logs_dir / "signal_history.jsonl")
    signals = list(_read_jsonl(sig_path))

    # In demo/CI, ensure we have enough signals for sensible stats
    if ctx.force_demo or (ctx.demo_mode and len(signals) < len(ctx.symbols) * 2):
        synth = _synth_signals(ctx, need_per_symbol=2)
        # append to canonical location respecting env override
        if os.getenv("SIGNALS_FILE"):
            out = Path(os.getenv("SIGNALS_FILE"))
        else:
            out = ctx.logs_dir / "signal_history.jsonl"
        _fsync_text(out, "", mode="a")
        for r in synth:
            _fsync_text(out, json.dumps(r) + "\n", mode="a")
        signals = list(_read_jsonl(out))

    book = _build_price_book(ctx)
    trades, equity_curve = _simulate(ctx, signals, book)

    # Write logs
    trades_path = ctx.logs_dir / "trades.jsonl"
    equity_path = ctx.logs_dir / "equity_curve.jsonl"
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

    # Metrics
    eq_vals = np.array([eq for _, eq in equity_curve], dtype=float)
    rets = np.diff(eq_vals) / np.maximum(eq_vals[:-1], 1e-12) if len(eq_vals) > 1 else np.array([])
    metrics = compute_metrics(eq_vals, rets, trades)

    # Calculate signals per day for the entire period
    if len(trades) > 1:
        time_span = (trades[-1].ts_entry - trades[0].ts_entry).total_seconds()
        days = max(1.0, time_span / 86400.0)  # Convert seconds to days
        signals_per_day = len(trades) / days
    else:
        signals_per_day = 0.0

    # By-symbol metrics
    by_symbol: Dict[str, Dict[str, Any]] = {}
    for s in ctx.symbols:
        sym_trades = [t for t in trades if t.symbol == s]
        eq_s = ctx.capital
        eq_series_s = [eq_s]
        for t in sym_trades:
            eq_s += t.pnl
            eq_series_s.append(eq_s)
        eq_series_s = np.array(eq_series_s, dtype=float)
        rets_s = np.diff(eq_series_s) / np.maximum(eq_series_s[:-1], 1e-12) if len(eq_series_s) > 1 else np.array([])
        m_s = compute_metrics(eq_series_s, rets_s, sym_trades)

        # Calculate signals per day for this symbol
        if len(sym_trades) > 1:
            sym_time_span = (sym_trades[-1].ts_entry - sym_trades[0].ts_entry).total_seconds()
            sym_days = max(1.0, sym_time_span / 86400.0)
            sym_signals_per_day = len(sym_trades) / sym_days
        else:
            sym_signals_per_day = 0.0

        by_symbol[s] = {
            "trades": len(sym_trades),
            "signals_per_day": round(sym_signals_per_day, 2),
            "sharpe": m_s["sharpe"],
            "sortino": m_s["sortino"],
            "max_drawdown": m_s["max_drawdown"],
            "win_rate": m_s["win_rate"],
            "profit_factor": m_s["profit_factor"],
            "cagr": m_s.get("cagr", None),
        }

    agg = {
        "trades": len(trades),
        "signals_per_day": round(signals_per_day, 2),
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

    # Plots
    if len(equity_curve) > 0:
        ts = [t for t, _ in equity_curve]
        eq = [e for _, e in equity_curve]
        plt.figure()
        plt.plot(ts, eq)
        plt.title("Equity Curve"); plt.xlabel("Time"); plt.ylabel("Equity")
        plt.tight_layout(); plt.savefig(ctx.artifacts_dir / "perf_equity_curve.png"); plt.close()

        arr = np.array(eq, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / np.maximum(peak, 1e-12)
        plt.figure()
        plt.plot(ts, dd)
        plt.title("Drawdown"); plt.xlabel("Time"); plt.ylabel("Drawdown")
        plt.tight_layout(); plt.savefig(ctx.artifacts_dir / "perf_drawdown.png"); plt.close()

    if rets.size > 0:
        plt.figure()
        plt.hist(rets, bins=20)
        plt.title("Per-Step Returns Histogram")
        plt.xlabel("Return"); plt.ylabel("Frequency")
        plt.tight_layout(); plt.savefig(ctx.artifacts_dir / "perf_returns_hist.png"); plt.close()

    labels = list(by_symbol.keys())
    if labels:
        vals = [by_symbol[s]["profit_factor"] if by_symbol[s]["profit_factor"] is not None else 0.0 for s in labels]
        plt.figure()
        plt.bar(labels, vals)
        plt.title("Profit Factor by Symbol")
        plt.tight_layout(); plt.savefig(ctx.artifacts_dir / "perf_by_symbol_bar.png"); plt.close()

    return out


# =========================
# SHIM: PaperTrader adapter
# =========================
class PaperTrader:
    """
    Adapter to support `from scripts.perf.paper_trader import PaperTrader`
    and provide a `replay_shadow(...)` method that converts shadow inference
    rows into normalized "paper trade intents".

    This does NOT compute PnL — it just normalizes/filters shadow rows and
    writes:
      - logs/paper_trades.jsonl
      - models/paper_summary.json
    """

    def __init__(self, accept_reasons: Optional[Iterable[str]] = None, min_conf: float = 0.0):
        # Shadow rows to consider trade-like (tweak as you enrich shadow logging)
        self.accept_reasons = set(accept_reasons or {"shadow", "shadow_only_live_ml_candidate", "live_ml_executed"})
        self.min_conf = float(min_conf)

    def _iter_shadow(self, shadow_path: Path) -> Iterable[Dict[str, Any]]:
        if not shadow_path.exists() or shadow_path.stat().st_size == 0:
            return []
        with shadow_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    yield json.loads(s)
                except Exception:
                    continue

    def _norm_ts(self, ts: Any) -> str:
        try:
            if isinstance(ts, str) and "T" in ts:
                return ts
        except Exception:
            pass
        return _iso(_now_utc())

    def replay_shadow(
        self,
        shadow_path: Path = Path("logs/signal_inference_shadow.jsonl"),
        trades_out: Path = Path("logs/paper_trades.jsonl"),
        summary_out: Path = Path("models/paper_summary.json"),
    ) -> Dict[str, Any]:
        _ensure_dir(trades_out.parent)
        _ensure_dir(summary_out.parent)

        total = 0
        considered = 0
        written = 0
        per_symbol: Dict[str, Dict[str, int]] = {}

        # start fresh
        _fsync_text(trades_out, "", mode="w")

        for row in self._iter_shadow(shadow_path):
            total += 1
            reason = row.get("reason")
            if reason not in self.accept_reasons:
                continue

            direction = row.get("ml_dir")
            confidence = row.get("ml_conf")
            symbol = row.get("symbol")

            if direction not in {"long", "short"}:
                continue

            try:
                conf = float(confidence) if confidence is not None else None
            except Exception:
                conf = None

            if conf is None or conf < self.min_conf:
                continue

            considered += 1

            trade = {
                "ts": self._norm_ts(row.get("ts")),
                "symbol": symbol,
                "side": direction,
                "confidence": conf,
                "source": reason,
                "governance": row.get("gov", {}),
            }
            _fsync_text(trades_out, json.dumps(trade) + "\n", mode="a")
            written += 1

            if symbol:
                per_symbol.setdefault(symbol, {"candidates": 0, "written": 0})
                per_symbol[symbol]["candidates"] += 1
                per_symbol[symbol]["written"] += 1

        summary = {
            "shadow_path": str(shadow_path),
            "trades_out": str(trades_out),
            "total_rows": total,
            "considered_rows": considered,
            "written_trades": written,
            "per_symbol": per_symbol,
        }
        _fsync_text(summary_out, json.dumps(summary, indent=2) + "\n", mode="w")
        return summary