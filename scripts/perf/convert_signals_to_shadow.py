# scripts/perf/convert_signals_to_shadow.py
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

SHADOW_PATH = Path("logs/signal_inference_shadow.jsonl")
HISTORY_CANDIDATES = [Path("logs/signal_history.jsonl"), Path("logs/signals.jsonl")]

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _parse_ts(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def _read_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def _pick_history_file() -> Optional[Path]:
    for p in HISTORY_CANDIDATES:
        if p.exists() and p.stat().st_size > 0:
            return p
    return None

def run(symbols: List[str], lookback_h: int) -> Dict[str, Any]:
    SHADOW_PATH.parent.mkdir(parents=True, exist_ok=True)

    src = _pick_history_file()
    if not src:
        return {"source": None, "written": 0, "reason": "no_history_found"}

    cutoff = _utcnow() - timedelta(hours=max(1, lookback_h))
    syms = {s.strip().upper() for s in symbols if s.strip()}
    written = 0

    with SHADOW_PATH.open("a", encoding="utf-8") as out:
        for row in _read_jsonl(src):
            sym = str(row.get("symbol", "")).upper()
            if syms and sym not in syms:
                continue

            ts = _parse_ts(str(row.get("ts") or row.get("timestamp") or ""))
            if not ts or ts < cutoff:
                continue

            # Accept a few common shapes:
            # - {"direction":"long"/"short","confidence":0.xx}
            # - {"dir":...,"conf":...}
            # - fallback: infer direction from score/label if present
            direction = row.get("direction") or row.get("dir")
            conf = row.get("confidence") or row.get("conf")

            if direction not in ("long", "short"):
                # try to infer direction if you stored a score in [0,1]
                score = row.get("score")
                if isinstance(score, (int, float)):
                    direction = "long" if score >= 0.5 else "short"
                    conf = float(score)
                else:
                    # no usable info; skip
                    continue

            if not isinstance(conf, (int, float)):
                # try a label
                label = str(row.get("confidence_label", "")).lower()
                if label.startswith("high"):
                    conf = 0.75
                elif label.startswith("medium"):
                    conf = 0.5
                elif label.startswith("low"):
                    conf = 0.3
                else:
                    continue

            payload = {
                "symbol": sym,
                "reason": "from_signal_history",
                "ml_ok": True,                 # “ok” just means actionable for paper
                "ml_dir": direction,
                "ml_conf": float(conf),
                # keep governance neutral; the replayer can still apply thresholds later if needed
                "gov": {"conf_min": 0.0, "debounce_min": 0},
                "ts": ts.astimezone(timezone.utc).isoformat(),
            }
            out.write(json.dumps(payload) + "\n")
            written += 1

    return {"source": str(src), "written": written, "reason": "ok"}

if __name__ == "__main__":
    syms = [s.strip() for s in os.getenv("MW_SHADOW_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()]
    lookback_h = int(os.getenv("MW_SHADOW_LOOKBACK_H", "72"))
    res = run(syms, lookback_h)
    print("[convert_signals_to_shadow]", json.dumps(res))