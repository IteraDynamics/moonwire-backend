# scripts/summary_sections/signal_quality_per_version.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, json
from typing import List, Dict, Any, Tuple, Optional

from .common import SummaryContext, parse_ts, is_demo_mode

# ---------- helpers ----------
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

def _within(ts: Optional[datetime], lo: datetime) -> bool:
    return bool(ts and ts >= lo)

def _classify(f1: Optional[float], n_labels: int, min_labels: int) -> Tuple[str, str]:
    if n_labels < min_labels or f1 is None:
        return "Insufficient", "ℹ️"
    if f1 >= 0.75:
        return "Strong", "✅"
    if f1 >= 0.40:
        return "Mixed", "⚠️"
    return "Weak", "❌"

def _safe_div(n: float, d: float) -> Optional[float]:
    return (n / d) if d else None

def _round_or_none(x: Optional[float], k: int = 2) -> Optional[float]:
    return round(x, k) if x is not None else None

# ---------- core ----------
def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build a per-version signal quality block by joining label feedback to nearby triggers,
    grouping by model_version, and computing precision/recall/F1.
    - precision = TP / (TP + FP)
    - recall    = TP / (TP + FN)   (FN counted as True labels with NO matching trigger)
    - F1        = harmonic mean of P and R (when both defined)
    """
    # knobs
    window_h = int(os.getenv("MW_SIGNAL_WINDOW_H", "72"))
    join_min = int(os.getenv("MW_SIGNAL_JOIN_MIN", "5"))
    min_labels = int(os.getenv("MW_THRESHOLD_MIN_LABELS", "3"))  # reuse existing knob

    t_end = _now_utc()
    t_start = t_end - timedelta(hours=window_h)

    # cache & read
    trig_rows = ctx.caches.get("trigger_rows")
    if trig_rows is None:
        trig_rows = _read_jsonl(ctx.models_dir / "trigger_history.jsonl")
        ctx.caches["trigger_rows"] = trig_rows

    lab_rows = ctx.caches.get("label_rows")
    if lab_rows is None:
        lab_rows = _read_jsonl(ctx.models_dir / "label_feedback.jsonl")
        ctx.caches["label_rows"] = lab_rows

    # filter by window
    triggers: List[Dict[str, Any]] = []
    for r in trig_rows:
        ts = parse_ts(r.get("timestamp"))
        if _within(ts, t_start):
            rr = dict(r)
            rr["_ts"] = ts
            triggers.append(rr)

    labels: List[Dict[str, Any]] = []
    for r in lab_rows:
        ts = parse_ts(r.get("timestamp"))
        if _within(ts, t_start):
            rr = dict(r)
            rr["_ts"] = ts
            labels.append(rr)

    # index triggers by origin for quick nearest match
    by_origin: Dict[str, List[Dict[str, Any]]] = {}
    for tr in triggers:
        by_origin.setdefault(str(tr.get("origin") or "unknown"), []).append(tr)
    for lst in by_origin.values():
        lst.sort(key=lambda x: x["_ts"])

    def _nearest_trigger(origin: str, ts: datetime) -> Optional[Dict[str, Any]]:
        lst = by_origin.get(origin or "unknown") or []
        if not lst:
            return None
        # binary search-ish
        lo, hi = 0, len(lst) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if lst[mid]["_ts"] < ts:
                lo = mid + 1
            else:
                hi = mid
        # check a small window around 'lo'
        best = None
        best_dt = None
        for j in range(max(0, lo - 3), min(len(lst), lo + 4)):
            dt = abs((lst[j]["_ts"] - ts).total_seconds())
            if best is None or dt < best_dt:
                best, best_dt = lst[j], dt
        if best and (best_dt or 10**9) <= join_min * 60:
            return best
        return None

    # per-version tallies
    per_version: Dict[str, Dict[str, Any]] = {}

    def _ver_from_pair(lbl: Dict[str, Any], trig: Optional[Dict[str, Any]]) -> str:
        return str(
            lbl.get("model_version")
            or (trig.get("model_version") if trig else None)
            or "unknown"
        )

    # 1) count TP / FP using matched labels↔triggers
    for lb in labels:
        origin = str(lb.get("origin") or "unknown")
        m = _nearest_trigger(origin, lb["_ts"])
        ver = _ver_from_pair(lb, m)

        d = per_version.setdefault(ver, {"tp": 0, "fp": 0, "fn": 0, "labels": 0, "triggers": 0})
        if m is not None:
            # we only count labels when we matched a trigger
            d["labels"] += 1
            d["triggers"] += 1
            is_true = bool(lb.get("label"))
            if is_true:
                d["tp"] += 1
            else:
                d["fp"] += 1
        else:
            # unmatched True labels count as FN (missed by the model)
            if bool(lb.get("label")):
                d["fn"] += 1
            # unmatched False labels do not contribute

    # Build rows + compute metrics
    rows: List[Dict[str, Any]] = []
    for ver, c in per_version.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        labels_n = c["labels"]
        triggers_n = c["triggers"]
        p = _safe_div(tp, tp + fp)
        r = _safe_div(tp, tp + fn)
        f1 = None
        if p is not None and r is not None and (p + r) > 0:
            f1 = 2 * p * r / (p + r)

        cls, emoji = _classify(f1, labels_n, min_labels)

        rows.append({
            "version": ver,
            "triggers": triggers_n,
            "true": tp,
            "false": fp,
            "labels": labels_n,
            "precision": p,
            "recall": r,
            "f1": f1,
            "class": cls,
            "emoji": emoji,
            "demo": False,
        })

    # demo seeding if empty or too sparse
    demo_used = False
    if not rows or sum(r["labels"] for r in rows) < 2:
        demo_used = True
        rows = [
            {"version": "v0.5.9", "triggers": 10, "true": 7, "false": 3, "labels": 10,
             "precision": 0.70, "recall": 0.64, "f1": 0.67, "class": "Mixed", "emoji": "⚠️", "demo": True},
            {"version": "v0.5.8", "triggers": 8,  "true": 6, "false": 2, "labels": 8,
             "precision": 0.75, "recall": 0.75, "f1": 0.75, "class": "Strong", "emoji": "✅", "demo": True},
            {"version": "v0.5.7", "triggers": 6,  "true": 2, "false": 4, "labels": 6,
             "precision": 0.33, "recall": 0.50, "f1": 0.40, "class": "Mixed", "emoji": "⚠️", "demo": True},
        ]

    # sort: Strong → Mixed → Weak → Insufficient, then by labels desc
    order = {"Strong": 0, "Mixed": 1, "Weak": 2, "Insufficient": 3}
    rows.sort(key=lambda r: (order.get(r["class"], 9), -r["labels"], r["version"]))

    # persist artifact
    artifact = {
        "window_hours": window_h,
        "join_minutes": join_min,
        "generated_at": t_end.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "per_version": rows,
        "demo": demo_used,
    }
    (ctx.models_dir).mkdir(parents=True, exist_ok=True)
    out_path = ctx.models_dir / "signal_quality_per_version.json"
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # markdown
    hdr = f"### 🧪 Per-Version Signal Quality ({window_h}h)" + (" (demo)" if demo_used else "")
    md.append(hdr)
    if not rows:
        md.append("_no data_")
        return

    for r in rows:
        P = _round_or_none(r["precision"], 2)
        R = _round_or_none(r["recall"], 2)
        F1 = _round_or_none(r["f1"], 2)
        md.append(f"- `{r['version']}` → {r['emoji']} {r['class']} (F1={F1 if F1 is not None else 'n/a'}, "
                  f"P={P if P is not None else 'n/a'}, R={R if R is not None else 'n/a'}, "
                  f"n={r['labels']})")