# scripts/perf/replay_shadow_to_paper.py
from __future__ import annotations

import json, os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from scripts.perf.paper_trader import Ctx as PTX, run_paper_trader  # uses your existing paper_trader

SHADOW_PATH = Path(os.getenv("SHADOW_LOG", "logs/signal_inference_shadow.jsonl"))
OUT_TRADES  = Path(os.getenv("PAPER_TRADES_LOG", "logs/paper_trades.jsonl"))

DEFAULT_CONF_MIN = float(os.getenv("REPLAY_CONF_MIN", "0.60"))  # floor unless row.gov overrides
LOOKBACK_H = int(os.getenv("REPLAY_LOOKBACK_H", os.getenv("MW_PERF_LOOKBACK_H", "72")))
ALLOWED    = {s.strip().upper() for s in os.getenv("MW_PERF_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()}

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def _parse_ts(v) -> Optional[datetime]:
    try:
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None

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

def _conf_min_from_row(row: Dict[str, Any]) -> float:
    gov = row.get("gov") or {}
    try:
        return float(gov.get("conf_min", DEFAULT_CONF_MIN))
    except Exception:
        return DEFAULT_CONF_MIN

def _row_qualifies(row: Dict[str, Any], win_start: datetime) -> bool:
    sym = str(row.get("symbol", "")).upper()
    if sym not in ALLOWED:
        return False
    ts = _parse_ts(row.get("ts")) or _now_utc()
    if ts < win_start:
        return False
    if not bool(row.get("ml_ok", False)):
        return False
    conf = row.get("ml_conf")
    if not isinstance(conf, (int, float)):
        return False
    return float(conf) >= _conf_min_from_row(row)

def _to_signal(row: Dict[str, Any]) -> Dict[str, Any]:
    ts = _parse_ts(row.get("ts")) or _now_utc()
    sym = str(row.get("symbol")).upper()
    direction = str(row.get("ml_dir", "")).lower()
    conf = float(row.get("ml_conf", 0.0) or 0.0)
    return {
        "id": f"shadow_{ts.isoformat()}_{sym}_{direction}",
        "ts": ts.isoformat().replace("+00:00","Z"),
        "symbol": sym,
        "direction": direction if direction in {"long","short"} else "long",
        "confidence": conf,
        "price": None,
        "source": "shadow",
        "model_version": "bundle/current",
        "outcome": None,
    }

def replay() -> Dict[str, Any]:
    win_start = _now_utc() - timedelta(hours=LOOKBACK_H)

    rows = list(_read_jsonl(SHADOW_PATH))
    selected = [_to_signal(r) for r in rows if _row_qualifies(r, win_start)]

    sig_path = Path("logs") / "signal_history.jsonl"
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    with sig_path.open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")

    ctx = PTX()
    ctx.symbols = list(ALLOWED)
    ctx.lookback_h = LOOKBACK_H
    ctx.force_demo = False
    ctx.demo_mode = False
    out = run_paper_trader(ctx, mode="replay-shadow")

    summary = {
        "shadow_path": str(SHADOW_PATH),
        "trades_out": str(OUT_TRADES),
        "total_rows": len(rows),
        "considered_rows": sum(1 for r in rows if (t := _parse_ts(r.get("ts"))) and t >= win_start),
        "written_trades": out.get("aggregate", {}).get("trades", 0),
        "per_symbol": {k: v.get("trades", 0) for k, v in out.get("by_symbol", {}).items()},
    }
    print("[replay_shadow_to_paper]", json.dumps(summary, indent=2))
    return summary

if __name__ == "__main__":
    replay()