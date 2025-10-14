# scripts/summary_sections/signal_quality_per_origin.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Any
import os, json, bisect

from .common import SummaryContext, parse_ts, is_demo_mode  # noqa: F401 (is_demo_mode retained for parity)

_EMOJI = {
    "Strong": "✅",
    "Mixed":  "⚠️",
    "Weak":   "❌",
    "Info":   "ℹ️",
}

def _load_jsonl_safe(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
            s = ln.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
    except Exception:
        return []
    return out

def _classify(precision: float | None, min_labels_observed: int) -> Tuple[str, str]:
    """
    Return (class, emoji).
    If fewer than 2 matched labels were observed, mark as Insufficient data (Info).
    """
    if min_labels_observed < 2 or precision is None:
        return ("Info", _EMOJI["Info"])
    if precision >= 0.75:
        return ("Strong", _EMOJI["Strong"])
    if precision >= 0.40:
        return ("Mixed", _EMOJI["Mixed"])
    return ("Weak", _EMOJI["Weak"])

def _nearest_within(ts_list: List[datetime], target: datetime, max_delta_min: int, used: set[int] | None = None) -> int | None:
    """
    ts_list must be sorted ascending.
    Return index of the closest UNUSED timestamp within ±max_delta_min of target.
    Expands outward to next-nearest on both sides until a valid unused candidate is found.
    """
    if not ts_list:
        return None
    used = used or set()
    pos = bisect.bisect_left(ts_list, target)
    L = pos - 1
    R = pos
    max_delta = max_delta_min * 60.0

    # Expand outward, always picking the nearest unused candidate first
    while L >= 0 or R < len(ts_list):
        candL = None
        candR = None

        if L >= 0 and L not in used:
            dtL = abs((ts_list[L] - target).total_seconds())
            if dtL <= max_delta:
                candL = (dtL, L)

        if R < len(ts_list) and R not in used:
            dtR = abs((ts_list[R] - target).total_seconds())
            if dtR <= max_delta:
                candR = (dtR, R)

        if candL is None and candR is None:
            # No valid candidate at this expansion width; expand further
            L -= 1
            R += 1
            continue

        # Choose the nearest (ties arbitrary)
        choices = [c for c in (candL, candR) if c is not None]
        dt, idx = min(choices, key=lambda t: t[0])
        return idx

    return None

def _compute(ctx: SummaryContext) -> Dict[str, Any]:
    """
    Returns:
      {
        "window_hours": int,
        "join_minutes": int,
        "generated_at": iso,
        "per_origin": [
          {"origin": "reddit", "triggers": 10, "true": 6, "false": 2,
           "labels": 8, "precision": 0.75, "class": "Strong", "emoji": "✅", "demo": False},
          ...
        ],
        "demo": bool
      }
    """
    window_h = int(os.getenv("MW_SIGNAL_WINDOW_H", "72"))
    join_min = int(os.getenv("MW_SIGNAL_JOIN_MIN", "5"))

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_h)

    trig_path = ctx.models_dir / "trigger_history.jsonl"
    lab_path  = ctx.models_dir / "label_feedback.jsonl"

    H = _load_jsonl_safe(trig_path)
    F = _load_jsonl_safe(lab_path)

    # Filter & normalize times
    def _norm_ts(v) -> datetime | None:
        try:
            return parse_ts(v)
        except Exception:
            return None

    trig_by_origin: Dict[str, List[datetime]] = {}
    for r in H:
        ts = _norm_ts(r.get("timestamp"))
        if not ts or ts < cutoff:
            continue
        origin = str(r.get("origin", "unknown")).lower()
        decision = str(r.get("decision", "")).lower()
        triggered = (decision == "triggered") or (r.get("triggered") is True)
        if not triggered:
            continue
        trig_by_origin.setdefault(origin, []).append(ts)

    labels_by_origin: Dict[str, List[Tuple[datetime, bool]]] = {}
    for r in F:
        ts = _norm_ts(r.get("timestamp"))
        if not ts or ts < cutoff:
            continue
        origin = str(r.get("origin", "unknown")).lower()
        lbl = bool(r.get("label", False))
        labels_by_origin.setdefault(origin, []).append((ts, lbl))

    # Sort by time
    for o, lst in trig_by_origin.items():
        lst.sort()
    for o, lst in labels_by_origin.items():
        lst.sort(key=lambda t: t[0])

    # Per-origin stats
    results = []
    used_labels_idx: Dict[str, set] = {o: set() for o in labels_by_origin.keys()}

    for origin, trig_ts_list in trig_by_origin.items():
        total_triggers = len(trig_ts_list)
        lbl_list = labels_by_origin.get(origin, [])
        lbl_times = [t for (t, _v) in lbl_list]

        t_pos = 0
        f_neg = 0

        if lbl_list:
            for tts in trig_ts_list:
                i = _nearest_within(lbl_times, tts, join_min, used=used_labels_idx.get(origin))
                if i is None:
                    continue
                if origin not in used_labels_idx:
                    used_labels_idx[origin] = set()
                used_labels_idx[origin].add(i)
                is_true = bool(lbl_list[i][1])
                if is_true:
                    t_pos += 1
                else:
                    f_neg += 1

        labels_n = t_pos + f_neg
        precision = (t_pos / float(labels_n)) if labels_n > 0 else None
        klass, emoji = _classify(precision, labels_n)

        results.append({
            "origin": origin,
            "triggers": total_triggers,
            "true": t_pos,
            "false": f_neg,
            "labels": labels_n,
            "precision": (float(precision) if precision is not None else None),
            "class": klass,
            "emoji": emoji,
            "demo": False,
        })

    # Demo fallback if empty and in demo mode
    need_demo = (not results) and ctx.is_demo
    if need_demo:
        demo = [
            {"origin": "twitter",  "triggers": 8, "true": 6, "false": 2},
            {"origin": "reddit",   "triggers":10, "true": 5, "false": 5},
            {"origin": "rss_news", "triggers": 6, "true": 1, "false": 5},
        ]
        results = []
        for row in demo:
            t_pos, f_neg = row["true"], row["false"]
            labels_n = t_pos + f_neg
            precision = t_pos / float(labels_n) if labels_n > 0 else None
            klass, emoji = _classify(precision, labels_n)
            results.append({
                "origin": row["origin"],
                "triggers": row["triggers"],
                "true": t_pos,
                "false": f_neg,
                "labels": labels_n,
                "precision": precision,
                "class": klass,
                "emoji": emoji,
                "demo": True,
            })

    # Sort: Strong → Mixed → Weak → Info; then by precision desc, then origin
    rank = {"Strong": 0, "Mixed": 1, "Weak": 2, "Info": 3}
    def _sort_key(r):
        prec = -float(r["precision"]) if isinstance(r.get("precision"), (int, float)) else 1.0
        return (rank.get(r["class"], 9), prec, r["origin"])
    results.sort(key=_sort_key)

    out = {
        "window_hours": window_h,
        "join_minutes": join_min,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "per_origin": results,
        "demo": any(r.get("demo") for r in results) if results else False,
    }
    return out

def append(md: List[str], ctx: SummaryContext) -> None:
    data = _compute(ctx)

    # Persist JSON artifact
    out_json = ctx.models_dir / "signal_quality_per_origin.json"
    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Markdown block
    window_h = data["window_hours"]
    md.append(f"\n### 📡 Per-Origin Signal Quality (last {window_h}h)")
    rows = data.get("per_origin", []) or []
    if data.get("demo"):
        md.append("_(seeded demo data)_")

    if not rows:
        md.append("_No recent triggers/labels available._")
        return

    for r in rows:
        origin = r["origin"]
        klass  = r["class"]
        emoji  = r["emoji"]
        labels = r["labels"]
        prec   = r["precision"]
        if klass == "Info":
            md.append(f"- `{origin}` → {_EMOJI['Info']} Insufficient data (n={labels})")
        else:
            try:
                md.append(f"- `{origin}` → {emoji} {klass} (precision={prec:.2f}, n={labels})")
            except Exception:
                md.append(f"- `{origin}` → {emoji} {klass} (precision=n/a, n={labels})")
