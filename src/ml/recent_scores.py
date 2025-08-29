# src/ml/recent_scores.py
from __future__ import annotations
import json, os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np

from src.paths import MODELS_DIR

RECENT_SCORES_PATH = MODELS_DIR / "recent_scores.jsonl"

@dataclass
class RecentScore:
    ts: datetime
    origin: str
    proba: float

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def append_recent_score(origin: str, proba: float, ts: Optional[datetime] = None, path: Path = RECENT_SCORES_PATH) -> None:
    """Append one line to recent_scores.jsonl (best-effort)."""
    try:
        ts = ts or _now_utc()
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"timestamp": ts.isoformat(), "origin": str(origin or "unknown"), "proba": float(proba)}
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        # Never break inference if logging fails
        pass

def _parse_ts(v) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        try:
            return datetime.fromtimestamp(float(v), tz=timezone.utc)
        except Exception:
            return None

def load_recent_scores(path: Path = RECENT_SCORES_PATH) -> List[RecentScore]:
    if not path.exists():
        return []
    out: List[RecentScore] = []
    try:
        for ln in path.read_text().splitlines():
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            ts = _parse_ts(obj.get("timestamp"))
            if not ts:
                continue
            try:
                p = float(obj.get("proba", 0.0))
            except Exception:
                p = 0.0
            out.append(RecentScore(ts=ts, origin=str(obj.get("origin", "unknown")), proba=p))
    except Exception:
        return []
    return out

def dynamic_threshold_for_origin(
    origin: str,
    *,
    window_hours: int = 48,
    min_samples: int = 30,
    quantile: float = None,
    fallback_static: float = None,
    recent: Optional[List[RecentScore]] = None,
) -> Tuple[Optional[float], int, float]:
    """
    Returns (dyn_threshold or None, n_recent, static_threshold_used)
    - If <min_samples> in window, returns (None, n_recent, static).
    - quantile: if None, read TL_DYN_THR_Q env (default 0.9).
    - fallback_static: if None, read TL_STATIC_PROBA_THR env (default 0.5).
    """
    if quantile is None:
        try:
            quantile = float(os.getenv("TL_DYN_THR_Q", "0.9"))
        except Exception:
            quantile = 0.9
    if fallback_static is None:
        try:
            fallback_static = float(os.getenv("TL_STATIC_PROBA_THR", "0.5"))
        except Exception:
            fallback_static = 0.5

    recent = recent or load_recent_scores()
    cutoff = _now_utc() - timedelta(hours=window_hours)
    vals = [r.proba for r in recent if r.origin == origin and r.ts >= cutoff]
    n = len(vals)
    if n >= min_samples:
        try:
            dyn = float(np.quantile(np.asarray(vals, dtype=float), quantile))
        except Exception:
            dyn = None
        return dyn, n, fallback_static
    else:
        return None, n, fallback_static
