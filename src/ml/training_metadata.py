from __future__ import annotations

import json, os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.paths import MODELS_DIR


RUNS_PATH = MODELS_DIR / "training_runs.jsonl"


@dataclass
class TrainingRun:
    timestamp: str
    version: str
    rows: int
    origin_counts: Dict[str, int]
    label_counts: Dict[str, int]
    metrics: Dict[str, Dict[str, float]]
    top_features: List[str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n")


def _iter_jsonl_reverse(path: Path):
    """
    Efficient-enough reverse iterator over JSONL lines for small/medium files.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    for ln in reversed(lines):
        s = (ln or "").strip()
        if not s:
            continue
        try:
            yield json.loads(s)
        except Exception:
            continue


def save_training_metadata(
    version: str,
    rows: int,
    origin_counts: Dict[str, int],
    label_counts: Dict[str, int],
    metrics: Dict[str, Dict[str, float]],
    top_features: List[str],
    *,
    timestamp: Optional[str] = None,
    runs_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Append one structured training-run record to models/training_runs.jsonl.
    """
    path = runs_path or RUNS_PATH
    rec = TrainingRun(
        timestamp=timestamp or _now_iso(),
        version=version,
        rows=int(rows),
        origin_counts={k: int(v) for k, v in (origin_counts or {}).items()},
        label_counts={k: int(v) for k, v in (label_counts or {}).items()},
        metrics={m: {mk: float(mv) for mk, mv in md.items()} for m, md in (metrics or {}).items()},
        top_features=list(top_features or []),
    )
    obj = asdict(rec)
    _append_jsonl(path, obj)
    return obj


def _seed_demo_if_needed(path: Path) -> Optional[Dict[str, Any]]:
    """
    If DEMO_MODE is on and the file is empty/missing, seed one plausible entry.
    """
    demo_on = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")
    if not demo_on:
        return None
    exists_and_nonempty = path.exists() and any((ln.strip() for ln in path.read_text().splitlines()))
    if exists_and_nonempty:
        return None

    demo = {
        "timestamp": _now_iso(),
        "version": "v0.5.1-demo",
        "rows": 48,
        "origin_counts": {"reddit": 18, "twitter": 16, "rss_news": 14},
        "label_counts": {"true": 30, "false": 18},
        "metrics": {
            "logistic": {"roc_auc": 0.91, "pr_auc": 0.88, "logloss": 0.42},
            "rf": {"roc_auc": 0.93, "pr_auc": 0.90, "logloss": 0.39},
            "gb": {"roc_auc": 0.92, "pr_auc": 0.89, "logloss": 0.40},
        },
        "top_features": ["count_24h", "burst_z", "precision_7d"],
    }
    _append_jsonl(path, demo)
    return demo


def load_latest_training_metadata(
    *, runs_path: Optional[Path] = None, allow_demo_seed: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Return the most recent record from models/training_runs.jsonl (or None).
    In DEMO_MODE (and allow_demo_seed=True), seeds a plausible record if empty.
    """
    path = runs_path or RUNS_PATH
    if allow_demo_seed:
        _seed_demo_if_needed(path)

    for obj in _iter_jsonl_reverse(path):
        # Basic sanity
        if isinstance(obj, dict) and "version" in obj and "metrics" in obj:
            return obj
    return None
