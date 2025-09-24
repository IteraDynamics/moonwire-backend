from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --- Small utils ----------------------------------------------------------------

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(ISO_FMT)


def _parse_ts(s: str) -> datetime:
    # Accept both '...Z' and general ISO with offsets
    try:
        if s.endswith("Z"):
            return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)
        x = datetime.fromisoformat(s)
        if x.tzinfo is None:
            x = x.replace(tzinfo=timezone.utc)
        return x.astimezone(timezone.utc)
    except Exception:
        # Best-effort: treat as UTC naive
        return datetime.fromisoformat(s.replace("Z", "")).replace(tzinfo=timezone.utc)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)


# --- Metrics --------------------------------------------------------------------

@dataclass
class BucketStats:
    bucket_start: datetime
    dim_value: str  # origin or version
    n: int
    ece: float
    brier: float


def _ece(scores: List[float], labels: List[bool], bins: int = 10) -> float:
    if not scores:
        return 0.0
    counts = [0] * bins
    conf_sum = [0.0] * bins
    acc_sum = [0.0] * bins
    for s, y in zip(scores, labels):
        s_clamped = min(0.999999, max(0.0, float(s)))
        idx = min(bins - 1, int(s_clamped * bins))
        counts[idx] += 1
        conf_sum[idx] += s_clamped
        acc_sum[idx] += 1.0 if y else 0.0
    total = len(scores)
    ece = 0.0
    for k in range(bins):
        if counts[k] == 0:
            continue
        avg_conf = conf_sum[k] / counts[k]
        avg_acc = acc_sum[k] / counts[k]
        ece += (counts[k] / total) * abs(avg_acc - avg_conf)
    return float(ece)


def _brier(scores: List[float], labels: List[bool]) -> float:
    if not scores:
        return 0.0
    return float(sum((float(s) - (1.0 if y else 0.0)) ** 2 for s, y in zip(scores, labels)) / len(scores))


# --- Core computation ------------------------------------------------------------

def _bucket_floor(ts: datetime, bucket_minutes: int) -> datetime:
    mins = (ts.minute // bucket_minutes) * bucket_minutes
    return ts.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=mins)


def _join_triggers_labels(
    triggers: Iterable[Dict[str, Any]],
    labels: Iterable[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    lab_by_id: Dict[str, Dict[str, Any]] = {r["id"]: r for r in labels if "id" in r}
    out: Dict[str, Dict[str, Any]] = {}
    for t in triggers:
        tid = t.get("id")
        if not tid:
            continue
        if tid in lab_by_id:
            out[tid] = {
                "id": tid,
                "timestamp": t.get("timestamp"),
                "origin": t.get("origin"),
                "version": t.get("version"),
                "score": float(t.get("score", 0.0)),
                "label": bool(lab_by_id[tid].get("label")),
            }
    return out


def _compute_trend(
    joined: Dict[str, Dict[str, Any]],
    dim: str,
    window_h: int,
    bucket_min: int,
    bins: int = 10,
) -> List[BucketStats]:
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(hours=window_h)

    groups: Dict[Tuple[datetime, str], List[Tuple[float, bool]]] = defaultdict(list)
    for r in joined.values():
        ts = _parse_ts(r["timestamp"])
        if ts < start_time or ts > now:
            continue
        dim_val = str(r.get(dim) or "unknown")
        bstart = _bucket_floor(ts, bucket_min)
        groups[(bstart, dim_val)].append((float(r["score"]), bool(r["label"])))

    stats: List[BucketStats] = []
    for (bstart, dim_val), pairs in sorted(groups.items(), key=lambda kv: kv[0][0]):
        scores = [s for s, _ in pairs]
        labels = [y for _, y in pairs]
        e = _ece(scores, labels, bins=bins)
        b = _brier(scores, labels)
        stats.append(BucketStats(bucket_start=bstart, dim_value=dim_val, n=len(pairs), ece=e, brier=b))
    return stats


# --- Public API (used by tests) -------------------------------------------------

def append(md: List[str], ctx) -> None:
    """
    Build calibration trend stats and emit markdown summary lines.
    Writes both:
      - models/calibration_reliability_trend.json (expected by tests)
      - models/calibration_trend.json            (back-compat)
    """
    models_dir = Path(getattr(ctx, "models_dir", "models"))
    logs_dir = Path(getattr(ctx, "logs_dir", "logs"))
    artifacts_dir = Path(getattr(ctx, "artifacts_dir", "artifacts"))
    _ensure_dir(models_dir)
    _ensure_dir(logs_dir)
    _ensure_dir(artifacts_dir)

    window_h = int(os.getenv("MW_CAL_TREND_WINDOW_H", "72"))
    bucket_min = int(os.getenv("MW_CAL_TREND_BUCKET_MIN", "120"))  # 2h default
    dim = os.getenv("MW_CAL_TREND_DIM", "origin")
    ece_bins = int(os.getenv("MW_CAL_ECE_BINS", "10"))

    trig = _read_jsonl(logs_dir / "trigger_history.jsonl")
    labs = _read_jsonl(logs_dir / "label_feedback.jsonl")
    joined = _join_triggers_labels(trig, labs)

    stats = _compute_trend(joined, dim=dim, window_h=window_h, bucket_min=bucket_min, bins=ece_bins)

    # Build series list
    series: List[Dict[str, Any]] = []
    for s in stats:
        series.append(
            {
                "bucket_start": _iso(s.bucket_start),
                "dim": dim,
                "value": s.dim_value,
                "n": s.n,
                "ece": s.ece,
                "brier": s.brier,
            }
        )

    # Wrap in object with "series" (what tests expect) + some meta
    obj = {
        "series": series,
        "meta": {
            "dim": dim,
            "window_h": window_h,
            "bucket_min": bucket_min,
            "ece_bins": ece_bins,
            "generated_at": _iso(datetime.now(timezone.utc)),
        },
    }

    # primary filename expected by tests
    trend_json = models_dir / "calibration_reliability_trend.json"
    trend_json.write_text(json.dumps(obj, indent=2))

    # back-compat filename used earlier in the work
    (models_dir / "calibration_trend.json").write_text(json.dumps(obj, indent=2))

    # Markdown
    md.append("### 🧮 Calibration & Reliability Trend vs Market Regimes (72h)")

    if not stats:
        present = sorted({str(r.get(dim) or "unknown") for r in trig})
        if present:
            for v in present:
                md.append(f"{v}  → (no joined labels in window)")
        else:
            md.append("_no data available_")
        return

    latest_by_val: Dict[str, BucketStats] = {}
    for s in stats:
        cur = latest_by_val.get(s.dim_value)
        if (cur is None) or (s.bucket_start > cur.bucket_start):
            latest_by_val[s.dim_value] = s

    for val, s in sorted(latest_by_val.items(), key=lambda kv: kv[0].lower()):
        ece_str = f"{s.ece:.3f}"
        brier_str = f"{s.brier:.3f}"
        md.append(f"{val}  → ECE {ece_str} | Brier {brier_str} | n {s.n}")