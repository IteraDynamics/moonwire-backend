# scripts/perf/replay_shadow_to_paper.py
from __future__ import annotations
import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

from scripts.perf.paper_trader import Ctx, run_paper_trader

ROOT = Path(".")
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

SHADOW = LOGS / "signal_inference_shadow.jsonl"
SIGNALS = LOGS / "signal_history.jsonl"   # what paper_trader reads

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00","Z")

def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out

def _append_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def _shadow_to_signals(rows: List[Dict[str, Any]], conf_min: float = 0.0) -> List[Dict[str, Any]]:
    """
    Map shadow rows -> minimal signal rows the paper trader understands.
    Fields used by paper_trader: ts, symbol, direction, (confidence optional).
    """
    out: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    # space rows one minute apart (stable ordering)
    step = timedelta(minutes=1)
    for idx, r in enumerate(rows):
        sym = str(r.get("symbol","")).upper()
        dir_ = r.get("ml_dir")
        conf = r.get("ml_conf")
        if not sym or dir_ not in {"long","short"}:
            continue
        try:
            c = float(conf) if conf is not None else 0.0
        except Exception:
            c = 0.0
        if c < conf_min:
            continue

        ts = r.get("ts")
        if ts:
            ts_str = str(ts)
            # normalize a bit
            if ts_str.endswith("Z"):
                ts_iso = ts_str
            else:
                # best-effort ISO passthrough
                ts_iso = ts_str
        else:
            ts_iso = (now + idx * step).replace(microsecond=0).isoformat().replace("+00:00","Z")

        out.append({
            "id": f"shadow_{sym}_{ts_iso}",
            "ts": ts_iso,
            "symbol": sym,
            "direction": dir_,
            "confidence": float(c),
            "source": "shadow",
            "model_version": "v0.9.2-shadow",
        })
    return out

def main():
    shadow_rows = _read_jsonl(SHADOW)
    # keep only recent subset (last ~500) to avoid huge appends
    shadow_rows = shadow_rows[-500:]

    # IMPORTANT: only convert actual ML rows
    # Filter out probe rows that have ml_ok false or missing ml_dir
    ml_rows = [r for r in shadow_rows if r.get("ml_ok") and r.get("ml_dir") in ("long","short")]

    # confidence gate = 0.0 so you can see trades immediately; bump later if desired
    signals = _shadow_to_signals(ml_rows, conf_min=float(os.getenv("REPLAY_CONF_MIN", "0.0")))
    if not signals:
        print(json.dumps({
            "shadow_path": str(SHADOW),
            "trades_out": str(LOGS / "paper_trades.jsonl"),
            "total_rows": len(shadow_rows),
            "considered_rows": len(ml_rows),
            "written_trades": 0,
            "per_symbol": {},
            "note": "No eligible ML rows; check ml_ok/dir/conf or relax REPLAY_CONF_MIN"
        }, indent=2))
        return

    # Append to signals file (paper_trader will look here)
    _append_jsonl(SIGNALS, signals)

    # Run paper trader over the last N hours (env overrides apply)
    ctx = Ctx()
    res = run_paper_trader(ctx, mode="replay-shadow")

    # Minimal summary to stdout (your workflow prints this in logs)
    print(json.dumps({
        "shadow_path": str(SHADOW),
        "trades_out": str(LOGS / "paper_trades.jsonl"),
        "total_rows": len(shadow_rows),
        "considered_rows": len(ml_rows),
        "written_trades": len(signals),
        "per_symbol": {s["symbol"]: None for s in signals},
        "metrics": res.get("aggregate", {}),
    }, indent=2))

if __name__ == "__main__":
    main()