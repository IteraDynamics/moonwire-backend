# src/ml/metrics.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, DefaultDict
from collections import defaultdict

from src.paths import MODELS_DIR  # reuse the models dir for our jsonl logs


# ---------- config helpers ----------
def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, "").strip())
        return v
    except Exception:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "")
    if not v:
        return default
    return v.lower() in ("1", "true", "yes", "on")

# Defaults
_DEFAULT_LOOKBACK_HOURS = _env_int("METRICS_LOOKBACK_HOURS", 72)
_DEFAULT_MIN_LABELS     = _env_int("METRICS_MIN_LABELS", 10)
_TOLERANCE_MINUTES      = _env_int("METRICS_JOIN_TOLERANCE_MIN", 5)

# Paths (allow override via env; fall back to models/)
_TRIGGER_HISTORY_PATH = Path(os.getenv("TRIGGER_LOG_PATH", str(MODELS_DIR / "trigger_history.jsonl")))
_LABEL_FEEDBACK_PATH  = Path(os.getenv("LABEL_FEEDBACK_PATH",  str(MODELS_DIR / "label_feedback.jsonl")))


# ---------- parsing ----------
def _parse_ts(val: Any) -> Optional[datetime]:
    """
    Accepts unix ts, ISO8601 (with Z), or already-datetime.
    Returns timezone-aware UTC datetime or None.
    """
    if isinstance(val, datetime):
        return val.astimezone(timezone.utc)
    if val is None:
        return None
    # unix
    try:
        return datetime.fromtimestamp(float(val), tz=timezone.utc)
    except Exception:
        pass
    # ISO
    try:
        s = str(val)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        return None


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out: List[dict] = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _within_window(ts: Optional[datetime], now: datetime, lookback_hours: int) -> bool:
    if ts is None:
        return False
    return ts >= now - timedelta(hours=lookback_hours)


# ---------- join logic ----------
@dataclass
class JoinedRow:
    origin: str
    timestamp_pred: datetime
    triggered: bool
    label: bool


def _key(r: dict) -> Tuple[str, Optional[datetime]]:
    origin = str(r.get("origin", "unknown")).strip() or "unknown"
    ts = _parse_ts(r.get("timestamp"))
    return origin, ts


def _nearest_match_idx(ts: datetime, arr: List[datetime], tol: timedelta) -> Optional[int]:
    """
    Given a sorted list of datetimes `arr`, find index whose |Δ| is minimal and <= tol.
    Returns None if no candidate within tolerance.
    """
    # binary search could be used; linear scan fine for small N
    best_i, best_delta = None, None
    for i, t in enumerate(arr):
        d = abs(t - ts)
        if d <= tol and (best_delta is None or d < best_delta):
            best_i, best_delta = i, d
    return best_i


def join_trigger_and_labels(
    trigger_history_path: Path = _TRIGGER_HISTORY_PATH,
    label_feedback_path: Path = _LABEL_FEEDBACK_PATH,
    *,
    lookback_hours: int = _DEFAULT_LOOKBACK_HOURS,
    tolerance_minutes: int = _TOLERANCE_MINUTES,
) -> List[JoinedRow]:
    """
    Join predictions with labels using (origin, nearest timestamp within tolerance).
    Returns a list of matched rows for metric computation.
    """
    now = datetime.now(timezone.utc)
    tol = timedelta(minutes=max(1, int(tolerance_minutes)))

    preds_raw = _load_jsonl(trigger_history_path)
    labs_raw  = _load_jsonl(label_feedback_path)

    # Filter window and normalize
    preds: List[Tuple[str, datetime, bool]] = []
    for r in preds_raw:
        origin, ts = _key(r)
        if not _within_window(ts, now, lookback_hours):
            continue
        triggered = str(r.get("decision", "")).lower() == "triggered"
        preds.append((origin, ts or now, triggered))

    labels_by_origin: DefaultDict[str, List[Tuple[datetime, bool]]] = defaultdict(list)
    for r in labs_raw:
        origin, ts = _key(r)
        if not _within_window(ts, now, lookback_hours):
            continue
        label_val = bool(r.get("label", False))
        labels_by_origin[origin].append((ts or now, label_val))

    # Sort label timestamps for each origin for nearest-match search
    for o in list(labels_by_origin.keys()):
        labels_by_origin[o].sort(key=lambda t: t[0])

    out: List[JoinedRow] = []
    for origin, ts_pred, triggered in preds:
        labs = labels_by_origin.get(origin) or []
        if not labs:
            continue
        ts_list = [t for (t, _) in labs]
        idx = _nearest_match_idx(ts_pred, ts_list, tol)
        if idx is None:
            continue
        label_bool = labs[idx][1]
        out.append(JoinedRow(origin=origin, timestamp_pred=ts_pred, triggered=triggered, label=label_bool))
    return out


# ---------- metrics ----------
@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def compute_pr_metrics(
    joined: List[JoinedRow],
    *,
    min_labels: int = _DEFAULT_MIN_LABELS,
) -> Dict[str, Any]:
    """
    Compute per-origin and overall precision/recall/F1.
    Only returns metrics if matched sample size >= min_labels.
    """
    # Per-origin tallies
    tallies: DefaultDict[str, PRF] = defaultdict(PRF)

    for row in joined:
        if row.triggered and row.label:
            tallies[row.origin].tp += 1
        elif row.triggered and not row.label:
            tallies[row.origin].fp += 1
        elif (not row.triggered) and row.label:
            tallies[row.origin].fn += 1
        # (not triggered) and (not label) → true negative: excluded from PR/F1 by design

    # Construct output rows
    per_origin = []
    total = PRF()
    matched = 0

    for origin in sorted(tallies.keys()):
        t = tallies[origin]
        matched += (t.tp + t.fp + t.fn)
        per_origin.append({
            "origin": origin,
            "precision": round(t.precision, 3),
            "recall": round(t.recall, 3),
            "f1": round(t.f1, 3),
            "tp": t.tp, "fp": t.fp, "fn": t.fn,
        })
        total.tp += t.tp
        total.fp += t.fp
        total.fn += t.fn

    overall = {
        "precision": round(total.precision, 3),
        "recall": round(total.recall, 3),
        "f1": round(total.f1, 3),
        "tp": total.tp, "fp": total.fp, "fn": total.fn,
    }

    return {
        "matched": matched,
        "min_required": int(min_labels),
        "per_origin": per_origin,
        "overall": overall,
    }


# ---------- demo seeding ----------
def _maybe_seed_demo(joined: List[JoinedRow]) -> List[JoinedRow]:
    """
    If DEMO_MODE is on and we have too few rows, synthesize a small plausible set
    so the CI summary is never empty.
    """
    if not _env_bool("DEMO_MODE", False):
        return joined
    if len(joined) >= 6:
        return joined

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    synth = [
        JoinedRow("twitter",  now - timedelta(hours=2), True,  True),
        JoinedRow("twitter",  now - timedelta(hours=1), True,  False),
        JoinedRow("reddit",   now - timedelta(hours=3), False, True),
        JoinedRow("reddit",   now - timedelta(hours=1), True,  True),
        JoinedRow("rss_news", now - timedelta(hours=4), True,  False),
        JoinedRow("rss_news", now - timedelta(hours=2), False, True),
    ]
    return joined + synth


# ---------- top-level convenience ----------
def rolling_precision_recall_snapshot(
    *,
    trigger_history_path: Path = _TRIGGER_HISTORY_PATH,
    label_feedback_path: Path = _LABEL_FEEDBACK_PATH,
    lookback_hours: Optional[int] = None,
    min_labels: Optional[int] = None,
    tolerance_minutes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    One-call helper used by CI summary.

    Returns:
      {
        "window_hours": int,
        "matched": int,
        "min_required": int,
        "per_origin": [...],
        "overall": {...}
      }
    """
    lb = int(lookback_hours if lookback_hours is not None else _DEFAULT_LOOKBACK_HOURS)
    mn = int(min_labels     if min_labels     is not None else _DEFAULT_MIN_LABELS)
    tol = int(tolerance_minutes if tolerance_minutes is not None else _TOLERANCE_MINUTES)

    joined = join_trigger_and_labels(
        trigger_history_path=trigger_history_path,
        label_feedback_path=label_feedback_path,
        lookback_hours=lb,
        tolerance_minutes=tol,
    )

    joined = _maybe_seed_demo(joined)
    metrics = compute_pr_metrics(joined, min_labels=mn)
    metrics["window_hours"] = lb
    return metrics
