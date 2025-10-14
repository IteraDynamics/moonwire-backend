# scripts/summary_sections/calibration_reliability.py
from __future__ import annotations

import os, json, math, statistics
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from .common import SummaryContext, parse_ts, _iso, is_demo_mode


# ---------- helpers ----------

def _load_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _safe_div(a: float, b: float) -> float:
    if not b:
        return 0.0
    return a / b


def _nearest_join(
    labels: List[dict],
    triggers: List[dict],
    join_minutes: int,
) -> List[Tuple[dict, Optional[dict]]]:
    """
    For each label row, find the nearest trigger of the same origin within ±join_minutes.
    Returns pairs (label_row, trigger_row_or_None).
    """
    by_origin: Dict[str, List[dict]] = defaultdict(list)
    for t in triggers:
        o = t.get("origin")
        ts = parse_ts(t.get("timestamp"))
        if not o or ts is None:
            continue
        t["_ts"] = ts
        by_origin[o].append(t)
    for o in by_origin:
        by_origin[o].sort(key=lambda r: r["_ts"])

    out: List[Tuple[dict, Optional[dict]]] = []
    max_delta = timedelta(minutes=join_minutes)
    for lab in labels:
        o = lab.get("origin")
        if not o:
            out.append((lab, None))
            continue
        Lts = parse_ts(lab.get("timestamp"))
        if Lts is None:
            out.append((lab, None))
            continue
        # binary-ish search over sorted list
        cand = None
        best = None
        rows = by_origin.get(o) or []
        # small list sizes in CI; linear scan is fine and robust
        for tr in rows:
            d = abs(tr["_ts"] - Lts)
            if d <= max_delta and (best is None or d < best):
                best = d
                cand = tr
        out.append((lab, cand))
    return out


def _ece_and_bins(y_prob: List[float], y_true: List[int], bins: int) -> Tuple[float, List[dict]]:
    """
    Equal-width bins over [0,1]. Returns (ECE, bins_list).
    bins_list entries: {"p_hat": mean_prob, "emp": mean_true, "n": count, "lo": lo, "hi": hi}
    """
    if not y_prob:
        return 0.0, []

    # Clamp to [0,1] defensively
    probs = [min(1.0, max(0.0, float(p))) for p in y_prob]
    trues = [1 if t else 0 for t in y_true]
    N = len(probs)

    edges = [i / bins for i in range(bins + 1)]
    per_bin_idx: List[List[int]] = [[] for _ in range(bins)]
    for i, p in enumerate(probs):
        # Put p==1.0 into last bin
        b = min(int(p * bins), bins - 1)
        per_bin_idx[b].append(i)

    out_bins = []
    ece = 0.0
    for b in range(bins):
        idxs = per_bin_idx[b]
        lo, hi = edges[b], edges[b + 1]
        if not idxs:
            out_bins.append({"p_hat": 0.0, "emp": 0.0, "n": 0, "lo": lo, "hi": hi})
            continue
        p_mean = statistics.fmean(probs[i] for i in idxs)
        t_mean = statistics.fmean(trues[i] for i in idxs)
        n_b = len(idxs)
        out_bins.append({"p_hat": p_mean, "emp": t_mean, "n": n_b, "lo": lo, "hi": hi})
        ece += (n_b / N) * abs(t_mean - p_mean)

    return float(ece), out_bins


def _brier(y_prob: List[float], y_true: List[int]) -> float:
    if not y_prob:
        return 0.0
    probs = [min(1.0, max(0.0, float(p))) for p in y_prob]
    trues = [1 if t else 0 for t in y_true]
    return float(statistics.fmean((p - t) ** 2 for p, t in zip(probs, trues)))


def _sanitize_version(s: str) -> str:
    if not s:
        return "unknown"
    return "".join(ch if ch.isalnum() or ch in ("_", ".", "-") else "_" for ch in s)


# ---------- main section ----------

def append(md: List[str], ctx: SummaryContext) -> None:
    models_dir = ctx.models_dir
    window_h = int(os.getenv("MW_CAL_WINDOW_H", "72"))
    join_min = int(os.getenv("MW_THRESHOLD_JOIN_MIN", "5"))
    bins = int(os.getenv("MW_CAL_BINS", "10"))
    min_labels = int(os.getenv("MW_CAL_MIN_LABELS", "50"))
    max_ece = float(os.getenv("MW_CAL_MAX_ECE", "0.06"))
    # per_origin toggle reserved for later polish; compute top-level per-version for now
    # per_origin = os.getenv("MW_CAL_PER_ORIGIN", "false").lower() in ("1","true","yes")

    now = datetime.now(timezone.utc)
    t_cut = now - timedelta(hours=window_h)

    # Load data
    trig_rows = _load_jsonl(models_dir / "trigger_history.jsonl")
    lab_rows = _load_jsonl(models_dir / "label_feedback.jsonl")

    # Filter by window and minimally valid fields
    trig_rows = [
        r for r in trig_rows
        if parse_ts(r.get("timestamp")) and parse_ts(r.get("timestamp")) >= t_cut
           and r.get("origin") and r.get("adjusted_score") is not None
    ]
    lab_rows = [
        r for r in lab_rows
        if parse_ts(r.get("timestamp")) and parse_ts(r.get("timestamp")) >= t_cut
           and r.get("origin") and (r.get("label") is True or r.get("label") is False)
    ]

    # Join
    joined = _nearest_join(lab_rows, trig_rows, join_min)

    # Group by model_version (prefer label.model_version else trigger.model_version)
    by_ver: Dict[str, Dict[str, List[Any]]] = defaultdict(lambda: {"y_prob": [], "y_true": []})
    for lab, trig in joined:
        ver = lab.get("model_version") or (trig or {}).get("model_version") or "unknown"
        if not trig:
            # if no matched trigger, we cannot assign a probability reliably
            continue
        p = float(trig.get("adjusted_score", 0.0))
        y = 1 if lab.get("label") is True else 0
        by_ver[ver]["y_prob"].append(p)
        by_ver[ver]["y_true"].append(y)

    # Demo seed if empty and demo mode
    demo_seeded = False
    if not by_ver and (ctx.is_demo or is_demo_mode()):
        demo_seeded = True
        # Seed two versions with plausible shapes
        # v_good: probs ~ [0.1]*m + [0.8]*m, labels aligned
        vg_prob = [0.1] * 12 + [0.8] * 12
        vg_true = [0] * 12 + [1] * 12
        by_ver["v_good"] = {"y_prob": vg_prob, "y_true": vg_true}
        # v_bad: probs ~0.9 but ~50% positives
        vb_prob = [0.9] * 24
        vb_true = [1 if i % 2 == 0 else 0 for i in range(24)]
        by_ver["v_bad"] = {"y_prob": vb_prob, "y_true": vb_true}

    # Compute metrics and plots
    per_version_rows: List[Dict[str, Any]] = []
    artifacts_dir = ctx.logs_dir.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    for ver, d in by_ver.items():
        y_prob = d["y_prob"]
        y_true = d["y_true"]
        n = len(y_prob)

        ece, bins_list = _ece_and_bins(y_prob, y_true, bins)
        brier = _brier(y_prob, y_true)

        alerts: List[str] = []
        if n < min_labels:
            alerts.append("low_n")
        elif ece > max_ece:
            alerts.append("high_ece")
        else:
            alerts.append("ok")

        per_version_rows.append({
            "version": ver,
            "ece": round(ece, 6),
            "brier": round(brier, 6),
            "n": int(n),
            "bins": [{"p_hat": round(b["p_hat"], 6), "emp": round(b["emp"], 6), "n": b["n"]} for b in bins_list],
            "alerts": alerts,
            "demo": bool(demo_seeded),
        })

        # Plot reliability curve for this version
        # Use only non-empty bins for markers; draw diagonal for reference.
        xs = [b["p_hat"] for b in bins_list if b["n"] > 0]
        ys = [b["emp"] for b in bins_list if b["n"] > 0]

        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)  # reference diagonal
        if xs:
            ax.plot(xs, ys, marker="o")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        title_demo = " (demo)" if demo_seeded else ""
        ax.set_title(f"Reliability — {ver}{title_demo}")
        ax.set_xlabel("Predicted probability (bin mean)")
        ax.set_ylabel("Empirical positive rate")

        fname = artifacts_dir / f"cal_reliability_{_sanitize_version(ver)}_.png"
        # match tests expecting 'cal_reliability_<version>.png'
        fname = artifacts_dir / f"cal_reliability_{_sanitize_version(ver)}.png"
        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)

    # Persist JSON
    out = {
        "window_hours": window_h,
        "bins": bins,
        "min_labels": min_labels,
        "max_ece": max_ece,
        "generated_at": _iso(now),
        "per_version": per_version_rows,
        "demo": bool(demo_seeded),
    }
    (models_dir / "calibration_reliability.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    # Markdown
    md.append(f"### 🧮 Calibration & Reliability ({window_h}h){' (demo)' if demo_seeded else ''}")
    if per_version_rows:
        for r in per_version_rows:
            ver = r["version"]
            md.append(
                f"- `{ver}` → ECE={r['ece']:.3f} | Brier={r['brier']:.3f} | n={r['n']} "
                f"[{','.join(r['alerts'])}]"
            )
    else:
        md.append("_no calibration data in window_")