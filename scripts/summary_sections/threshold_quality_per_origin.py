# scripts/summary_sections/threshold_quality_per_origin.py
from __future__ import annotations

import os, json, math
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any

# We rely on the orchestrator to pass SummaryContext, but keep this import-free here.

# ---------- small utils ----------
def _parse_ts(val) -> datetime | None:
    if val is None:
        return None
    try:
        # float epoch
        ts = float(val)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
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
    try:
        for ln in path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    except Exception:
        return []
    return out


def _within(a: datetime, b: datetime, minutes: int) -> bool:
    return abs((a - b).total_seconds()) <= minutes * 60


def _score_of(d: dict) -> float | None:
    for k in ("adjusted_score", "score", "prob", "proba"):
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                continue
    return None


def _prec(tp: int, fp: int) -> float | None:
    denom = tp + fp
    if denom <= 0:
        return 0.0
    return tp / denom


def _rec(tp: int, fn: int) -> float | None:
    denom = tp + fn
    if denom <= 0:
        return 0.0
    return tp / denom


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    # When both are zero, F1 should be 0.0 (not None)
    if p == 0.0 and r == 0.0:
        return 0.0
    denom = p + r
    if denom == 0.0:
        return 0.0
    return 2 * p * r / denom


@dataclass
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    threshold: float = 0.5

    def n(self) -> int:
        return self.tp + self.fp + self.fn

    def precision(self) -> float | None:
        return _prec(self.tp, self.fp)

    def recall(self) -> float | None:
        return _rec(self.tp, self.fn)

    def f1(self) -> float | None:
        return _f1(self.precision(), self.recall())


def _classify(f1: float | None, n: int, min_labels: int) -> Tuple[str, str]:
    """
    Returns (label, emoji)
    """
    if n < min_labels:
        return ("Insufficient", "ℹ️")
    v = 0.0 if f1 is None else float(f1)
    if v >= 0.75:
        return ("Strong", "✅")
    if v >= 0.40:
        return ("Mixed", "⚠️")
    return ("Weak", "❌")


def _fmt2(x: float | None) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "n/a"


def _load_thresholds(models_dir: Path) -> Dict[str, float]:
    p = models_dir / "per_origin_thresholds.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        out: Dict[str, float] = {}
        for k, v in (data or {}).items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out
    except Exception:
        return {}


def _group_by_origin(rows: List[dict]) -> Dict[str, List[dict]]:
    by: Dict[str, List[dict]] = {}
    for r in rows:
        o = r.get("origin", "unknown")
        by.setdefault(o, []).append(r)
    # Sort by timestamp ascending
    for o in by:
        by[o].sort(key=lambda d: (_parse_ts(d.get("timestamp")) or datetime.fromtimestamp(0, tz=timezone.utc)))
    return by


def _match_label_to_trigger(
    label_row: dict,
    trig_rows_sorted: List[dict],
    join_minutes: int
) -> dict | None:
    """
    Best-effort: pick the nearest trigger in time for the same origin within ±join_minutes.
    trig_rows_sorted is already sorted ascending by timestamp.
    """
    lt = _parse_ts(label_row.get("timestamp"))
    if lt is None:
        return None
    best = None
    best_dt = None
    for tr in trig_rows_sorted:
        tt = _parse_ts(tr.get("timestamp"))
        if tt is None:
            continue
        dt = abs((lt - tt).total_seconds())
        if dt <= join_minutes * 60:
            if best_dt is None or dt < best_dt:
                best = tr
                best_dt = dt
    return best


def _seed_demo() -> Dict[str, _Counts]:
    # Plausible seeded counts
    return {
        "twitter":  _Counts(tp=6, fp=2, fn=2, threshold=0.50),
        "reddit":   _Counts(tp=5, fp=5, fn=3, threshold=0.50),
        "rss_news": _Counts(tp=1, fp=5, fn=4, threshold=0.50),
    }


# ---------- public entry ----------
def append(md: List[str], ctx) -> None:
    """
    Build 'Per-Origin Threshold Quality (48h)' block.
    Inputs: models/trigger_history.jsonl, models/label_feedback.jsonl, models/per_origin_thresholds.json
    Output: markdown lines + models/threshold_quality_per_origin.json artifact.
    """
    # Knobs
    window_h = int(os.getenv("MW_SCORE_WINDOW_H", "48"))
    join_min = int(os.getenv("MW_THRESHOLD_JOIN_MIN", "5"))
    min_labels = int(os.getenv("MW_THRESHOLD_MIN_LABELS", "3"))
    now = datetime.now(timezone.utc)
    t_min = now - timedelta(hours=window_h)

    models_dir: Path = ctx.models_dir
    artifacts_dir: Path = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    # Cache-friendly loads
    trig_rows = ctx.caches.get("trigger_rows")
    if trig_rows is None:
        trig_rows = _load_jsonl(models_dir / "trigger_history.jsonl")
        ctx.caches["trigger_rows"] = trig_rows

    lab_rows = ctx.caches.get("label_rows")
    if lab_rows is None:
        lab_rows = _load_jsonl(models_dir / "label_feedback.jsonl")
        ctx.caches["label_rows"] = lab_rows

    # Filter window
    trig_recent = [r for r in trig_rows if (_parse_ts(r.get("timestamp")) or t_min) >= t_min]
    lab_recent  = [r for r in lab_rows  if (_parse_ts(r.get("timestamp")) or t_min) >= t_min]

    thresholds = _load_thresholds(models_dir)
    default_thr = 0.5

    # Group by origin for faster local matches
    trig_by_origin = _group_by_origin(trig_recent)
    # We'll walk labels and find nearest trigger by origin
    counts_by_origin: Dict[str, _Counts] = {}

    for lab in lab_recent:
        origin = lab.get("origin", "unknown")
        label_val = bool(lab.get("label", False))
        group = trig_by_origin.get(origin, [])
        if not group:
            continue
        tr = _match_label_to_trigger(lab, group, join_min)
        if tr is None:
            continue

        thr = thresholds.get(origin, default_thr)
        c = counts_by_origin.get(origin)
        if c is None:
            c = _Counts(threshold=thr)
            counts_by_origin[origin] = c

        # Update the threshold if file had a specific value (keep first seen)
        if origin in thresholds:
            c.threshold = thresholds[origin]

        sc = _score_of(tr)
        predicted = False
        try:
            predicted = (sc is not None) and (float(sc) >= float(c.threshold))
        except Exception:
            predicted = False

        if label_val and predicted:
            c.tp += 1
        elif (not label_val) and predicted:
            c.fp += 1
        elif label_val and (not predicted):
            c.fn += 1
        else:
            # label False and predicted False → true negative; we don't track TN for P/R/F1
            pass

    # Demo fallback: if no origins produced counts and we're in demo mode
    produced = len([o for o, c in counts_by_origin.items() if c.n() > 0])
    demo_used = False
    if produced == 0 and (os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes") or getattr(ctx, "is_demo", False)):
        counts_by_origin = _seed_demo()
        demo_used = True

    # Build rows
    rows = []
    for origin, c in counts_by_origin.items():
        p = c.precision()
        r = c.recall()
        f1 = c.f1()
        n = c.n()
        klass, emoji = _classify(f1, n, min_labels)
        rows.append({
            "origin": origin,
            "threshold": float(c.threshold),
            "tp": c.tp, "fp": c.fp, "fn": c.fn,
            "precision": (None if p is None else float(p)),
            "recall":    (None if r is None else float(r)),
            "f1":        (None if f1 is None else float(f1)),
            "class": klass,
            "emoji": emoji,
            "n": n,
        })

    # Sort: Strong → Mixed → Weak → Insufficient, then by origin
    order = {"Strong": 0, "Mixed": 1, "Weak": 2, "Insufficient": 3}
    rows.sort(key=lambda d: (order.get(d["class"], 9), d["origin"]))

    # Artifact write
    artifact = {
        "window_hours": window_h,
        "join_minutes": join_min,
        "min_labels": min_labels,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "per_origin": rows,
        "demo": bool(demo_used),
    }
    (models_dir / "threshold_quality_per_origin.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # Markdown
    head = f"### 📊 Per-Origin Threshold Quality ({window_h}h)"
    if demo_used:
        head += " (demo)"
    md.append(head)
    if not rows:
        md.append("_No matched label/trigger pairs in the window._")
        return

    for r in rows:
        origin = r["origin"]
        emoji = r["emoji"]
        klass = r["class"]
        f1 = _fmt2(r["f1"])
        p  = _fmt2(r["precision"])
        rr = _fmt2(r["recall"])
        n  = r["n"]
        thr = f"{float(r['threshold']):.2f}"
        md.append(f"- `{origin}` → {emoji} {klass} (F1={f1}, P={p}, R={rr}, n={n}, threshold={thr})")