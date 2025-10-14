# scripts/summary_sections/signal_quality_per_version.py
from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone, timedelta

# matplotlib headless
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from .common import SummaryContext, parse_ts, _iso

# ---------------------------- small helpers ----------------------------

def _safe_div(num: float, den: float) -> float:
    try:
        return float(num) / float(den) if den else 0.0
    except Exception:
        return 0.0


def _class_from_f1(f1: float, n_labels: int) -> Tuple[str, str]:
    """
    Buckets:
      - ✅ Strong (F1 ≥ 0.75)
      - ⚠️ Mixed (0.40 ≤ F1 < 0.75)
      - ❌ Weak  (F1 < 0.40)
      - ℹ️ Insufficient (labels < 2)
    """
    if n_labels < 2:
        return ("Insufficient", "ℹ️")
    if f1 >= 0.75:
        return ("Strong", "✅")
    if f1 >= 0.40:
        return ("Mixed", "⚠️")
    return ("Weak", "❌")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _nearest_join(
    labels: List[Dict[str, Any]],
    triggers: List[Dict[str, Any]],
    join_min: int,
) -> List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
    """
    For each label, find the nearest trigger on the same origin within ±join_min minutes.
    Returns list of (label_row, trigger_row_or_None).
    """
    by_origin: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = defaultdict(list)
    for t in triggers:
        o = t.get("origin") or "unknown"
        ts = parse_ts(t.get("timestamp"))
        if not ts:
            continue
        by_origin[o].append((ts, t))
    for o in by_origin:
        by_origin[o].sort(key=lambda x: x[0])

    out: List[Tuple[Dict[str, Any], Optional[Dict[str, Any]]]] = []
    window = timedelta(minutes=join_min)

    for lab in labels:
        o = lab.get("origin") or "unknown"
        lts = parse_ts(lab.get("timestamp"))
        if not lts:
            out.append((lab, None))
            continue

        best: Optional[Dict[str, Any]] = None
        best_dt = None
        rows = by_origin.get(o, [])
        for tts, row in rows:
            dt = abs(tts - lts)
            if dt <= window and (best_dt is None or dt < best_dt):
                best_dt = dt
                best = row
        out.append((lab, best))

    return out


def _maybe_seed_series_if_demo(
    series: List[Dict[str, Any]] | None,
    per_version: List[Dict[str, Any]],
    now: datetime,
    is_demo: bool,
) -> List[Dict[str, Any]]:
    if not is_demo:
        return series or []
    if series:
        return series

    versions = [r.get("version") for r in (per_version or []) if r.get("version")]
    if not versions:
        versions = ["v0.5.9", "v0.5.8"]

    seeds: List[Dict[str, Any]] = []
    for v in versions[:3]:
        pts = [
            (now - timedelta(hours=6), 0.62),
            (now - timedelta(hours=3), 0.69),
            (now,                     0.74),
        ]
        for t, p in pts:
            seeds.append({"version": v, "t": _iso(t), "precision": round(p, 2)})
    return seeds


# ---------------------------- main entrypoint ----------------------------

def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Renders:
      - Snapshot table for per-version signal quality (F1/P/R)
      - Precision trend chart across time per version (if series available)

    Reads/writes:
      - models/signal_quality_per_version.json
      - artifacts/signal_quality_by_version_{window}h.png
    """
    models_dir = ctx.models_dir
    window_h = int(os.getenv("MW_SIGNAL_WINDOW_H", "72"))
    join_min = int(os.getenv("MW_SIGNAL_JOIN_MIN", "5"))
    want_chart = os.getenv("MW_SIGNAL_VERSION_CHART", "true").lower() in ("1", "true", "yes")

    out_json = models_dir / "signal_quality_per_version.json"
    now = datetime.now(timezone.utc)

    # Try to load an existing artifact; if missing or incomplete, compute snapshot from raw logs
    data: Dict[str, Any] = {}
    if out_json.exists():
        try:
            data = json.loads(out_json.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}

    per_version = data.get("per_version")
    series = data.get("series")

    if not isinstance(per_version, list):
        trig_rows = _load_jsonl(models_dir / "trigger_history.jsonl")
        lab_rows = _load_jsonl(models_dir / "label_feedback.jsonl")

        cutoff = now - timedelta(hours=window_h)
        trig_rows = [r for r in trig_rows if (parse_ts(r.get("timestamp")) or now) >= cutoff]
        lab_rows  = [r for r in lab_rows  if (parse_ts(r.get("timestamp")) or now) >= cutoff]

        joined = _nearest_join(lab_rows, trig_rows, join_min)

        counts = defaultdict(lambda: {"true": 0, "false": 0, "labels": 0, "triggers": 0})
        for lab, trig in joined:
            version = lab.get("model_version") or (trig or {}).get("model_version") or "unknown"
            if trig is not None:
                counts[version]["triggers"] += 1
            if lab.get("label") is True:
                counts[version]["true"] += 1
                counts[version]["labels"] += 1
            elif lab.get("label") is False:
                counts[version]["false"] += 1
                counts[version]["labels"] += 1

        per_version = []
        for v, c in counts.items():
            tp = int(c["true"])
            fp = int(c["false"])
            fn = 0  # keep fn=0 here (only matched labels considered)
            P = _safe_div(tp, tp + fp)
            R = _safe_div(tp, tp + fn)
            F1 = _safe_div(2 * P * R, (P + R)) if (P + R) else 0.0
            klass, emoji = _class_from_f1(F1, int(c["labels"]))
            per_version.append({
                "version": v,
                "triggers": int(c["triggers"]),
                "true": tp,
                "false": fp,
                "labels": int(c["labels"]),
                "precision": round(P, 2),
                "recall": round(R, 2),
                "f1": round(F1, 2),
                "class": klass,
                "emoji": emoji,
                "demo": False,
            })

        order = {"Strong": 0, "Mixed": 1, "Weak": 2, "Insufficient": 3}
        per_version.sort(key=lambda r: (order.get(r["class"], 9), -r["f1"], r["version"]))

        if not per_version and ctx.is_demo:
            per_version = [
                {"version": "v0.5.9", "triggers": 8,  "true": 6, "false": 2, "labels": 8,
                 "precision": 0.75, "recall": 0.75, "f1": 0.75, "class": "Strong", "emoji": "✅", "demo": True},
                {"version": "v0.5.8", "triggers": 10, "true": 7, "false": 3, "labels": 10,
                 "precision": 0.70, "recall": 0.64, "f1": 0.67, "class": "Mixed",  "emoji": "⚠️", "demo": True},
            ]

        data = {
            "window_hours": window_h,
            "join_minutes": join_min,
            "generated_at": _iso(now),
            "per_version": per_version,
            "series": [],
            "demo": ctx.is_demo,
        }

    # -------- normalize per_version rows (fixes KeyError in tests) --------
    pv_list = data.get("per_version", []) or []
    for r in pv_list:
        P = float(r.get("precision", 0.0) or 0.0)
        R = float(r.get("recall", P))
        F1 = float(r.get("f1", 0.0) or 0.0)
        if F1 == 0.0 and (P or R):
            F1 = (2 * P * R / (P + R)) if (P + R) else P
        r["precision"] = round(P, 2)
        r["recall"]    = round(R, 2)
        r["f1"]        = round(F1, 2)

        labels = int(r.get("labels", 0) or 0)
        if "class" not in r or "emoji" not in r:
            klass, emoji = _class_from_f1(r["f1"], labels)
            r.setdefault("class", klass)
            r.setdefault("emoji", emoji)
    data["per_version"] = pv_list

    # -------- ensure / maybe seed series for chart --------
    if not isinstance(data.get("series"), list):
        data["series"] = []
    data["series"] = _maybe_seed_series_if_demo(
        data.get("series"), data.get("per_version") or [], now, data.get("demo", False)
    )

    # Persist JSON so both summary and tests read the same normalized artifact
    out_json.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    # ---------------------------- markdown ----------------------------
    md.append(f"### 🧪 Per-Version Signal Quality ({window_h}h){' (demo)' if data.get('demo') else ''}")
    pv = data.get("per_version") or []
    if pv:
        for r in pv:
            md.append(
                f"- `{r.get('version','unknown')}` → {r.get('emoji','ℹ️')} {r.get('class','Insufficient')} "
                f"(F1={float(r.get('f1',0.0)):.2f}, P={float(r.get('precision',0.0)):.2f}, "
                f"R={float(r.get('recall',0.0)):.2f}, n={int(r.get('labels',0))})"
            )
    else:
        md.append("_no per-version summary available_")

    # ---------------------------- chart ----------------------------
    chart_note = "no time series available"
    if want_chart:
        series = data.get("series") or []
        if series:
            by_v: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
            for row in series:
                v = row.get("version") or "unknown"
                t = parse_ts(row.get("t"))
                p = row.get("precision")
                if t is None or p is None:
                    continue
                by_v[v].append((t, float(p)))
            for v in by_v:
                by_v[v].sort(key=lambda x: x[0])

            plt.figure(figsize=(8, 3.5), dpi=150)
            ax = plt.gca()
            ax.axhspan(0.0, 0.40, alpha=0.10)
            ax.axhspan(0.40, 0.75, alpha=0.07)
            ax.axhspan(0.75, 1.00, alpha=0.10)

            for v, pts in by_v.items():
                xs = [t for (t, _) in pts]
                ys = [p for (_, p) in pts]
                if not xs:
                    continue
                plt.plot(xs, ys, marker="o", linewidth=1.5, label=v)

            plt.ylim(0.0, 1.0)
            plt.ylabel("Precision")
            plt.title(f"Per-Version Precision Trend ({window_h}h){' • demo' if data.get('demo') else ''}")
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout()

            # Save in the repo run’s artifacts directory (matches tests):
            # tests create tmp_path/artifacts and expect the file to land there.
            run_root = models_dir.resolve().parent
            artifacts_dir = run_root / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            img_path = artifacts_dir / f"signal_quality_by_version_{window_h}h.png"
            plt.savefig(img_path)
            plt.close()

            chart_note = str(img_path)
        else:
            chart_note = "no time series available (enable DEMO_MODE or accumulate runs)"

    md.append(f"\n📈 Per-Version Precision Trend: {chart_note}")