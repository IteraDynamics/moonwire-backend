# scripts/summary_sections/calibration_reliability.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple, DefaultDict
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import os, json, math

import matplotlib
matplotlib.use("Agg")  # headless for CI
import matplotlib.pyplot as plt  # noqa: E402

from .common import SummaryContext, parse_ts, _iso


# ---------- tiny io helpers (local; avoid coupling) ----------
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------- joins & math ----------
def _nearest_join_labels_to_triggers(
    labels: List[Dict[str, Any]],
    triggers: List[Dict[str, Any]],
    join_minutes: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any] | None]]:
    """
    For each label row, find the nearest trigger in the same origin within ±join_minutes.
    Return list of (label_row, trigger_row or None).
    """
    by_origin: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in triggers:
        o = t.get("origin") or "unknown"
        ts = parse_ts(t.get("timestamp"))
        if ts is None:
            continue
        t["_ts"] = ts
        by_origin[o].append(t)
    for o in by_origin:
        by_origin[o].sort(key=lambda r: r["_ts"])

    out: List[Tuple[Dict[str, Any], Dict[str, Any] | None]] = []
    max_delta = timedelta(minutes=join_minutes)

    for lab in labels:
        o = lab.get("origin") or "unknown"
        lts = parse_ts(lab.get("timestamp"))
        if lts is None:
            out.append((lab, None))
            continue
        lab["_ts"] = lts

        cand = by_origin.get(o) or []
        # binary search for insertion point
        lo, hi = 0, len(cand)
        while lo < hi:
            mid = (lo + hi) // 2
            if cand[mid]["_ts"] < lts:
                lo = mid + 1
            else:
                hi = mid
        # neighbors at lo-1 and lo
        best = None
        best_dt = None
        for idx in (lo - 1, lo):
            if 0 <= idx < len(cand):
                dt = abs(cand[idx]["_ts"] - lts)
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best = cand[idx]
        if best is not None and best_dt is not None and best_dt <= max_delta:
            out.append((lab, best))
        else:
            out.append((lab, None))
    return out


def _clip01(x: float | None) -> float | None:
    if x is None:
        return None
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return None


def _ece_and_bins(y_prob: List[float], y_true: List[int], bins: int) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Equal-width bins on [0,1]. Returns (ECE, bins_summary).
    bins_summary: list of {"p_hat":mean_prob,"emp":mean_true,"n":count}
    """
    n = len(y_prob)
    if n == 0:
        return 0.0, []

    # Pre-allocate containers
    sums_prob = [0.0] * bins
    sums_true = [0.0] * bins
    counts = [0] * bins

    for p, y in zip(y_prob, y_true):
        # Map 1.0 to last bin
        idx = min(bins - 1, int(p * bins))
        sums_prob[idx] += p
        sums_true[idx] += y
        counts[idx] += 1

    out_bins: List[Dict[str, Any]] = []
    ece = 0.0
    for i in range(bins):
        if counts[i] > 0:
            mean_p = sums_prob[i] / counts[i]
            emp = sums_true[i] / counts[i]
            out_bins.append({"p_hat": round(mean_p, 4), "emp": round(emp, 4), "n": int(counts[i])})
            ece += (counts[i] / n) * abs(emp - mean_p)
        else:
            # Represent empty bins so curves look continuous if needed (optional to include)
            out_bins.append({"p_hat": round((i + 0.5) / bins, 4), "emp": None, "n": 0})
    return float(ece), out_bins


def _brier(y_prob: List[float], y_true: List[int]) -> float:
    n = len(y_prob)
    if n == 0:
        return 0.0
    s = 0.0
    for p, y in zip(y_prob, y_true):
        d = (p - y)
        s += d * d
    return float(s / n)


def _classify_alerts(ece: float, n: int, min_labels: int, max_ece: float) -> List[str]:
    alerts: List[str] = []
    if n < min_labels:
        alerts.append("low_n")
    elif ece > max_ece:
        alerts.append("high_ece")
    else:
        alerts.append("ok")
    return alerts


# ---------- plotting ----------
def _plot_reliability(version: str, bins_data: List[Dict[str, Any]], ece: float, n: int, out_path: Path, is_demo: bool) -> None:
    xs, ys, ws = [], [], []
    for b in bins_data:
        ph = b.get("p_hat")
        emp = b.get("emp")
        if emp is None:
            continue
        xs.append(ph)
        ys.append(emp)
        ws.append(b.get("n", 0))

    _ensure_dir(out_path.parent)
    plt.figure(figsize=(6.4, 3.6), dpi=140)
    # diagonal
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    # curve
    if xs:
        plt.plot(xs, ys, marker="o")
    plt.xlabel("predicted probability (p̂)")
    plt.ylabel("empirical positive rate")
    title = f"Reliability — {version} (ECE={ece:.3f}, n={n})"
    if is_demo:
        title += " [demo]"
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- main entry ----------
def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Build per-version calibration metrics & reliability curves.
    Writes:
      - models/calibration_reliability.json
      - artifacts/cal_reliability_<version>.png (one per version)
    Appends markdown summary lines.
    """
    models_dir = ctx.models_dir
    logs_dir = ctx.logs_dir  # not used here, but ctx is consistent
    window_h = int(os.getenv("MW_CAL_WINDOW_H", "72"))
    join_min = int(os.getenv("MW_THRESHOLD_JOIN_MIN", "5"))
    n_bins = int(os.getenv("MW_CAL_BINS", "10"))
    min_labels = int(os.getenv("MW_CAL_MIN_LABELS", "50"))
    max_ece = float(os.getenv("MW_CAL_MAX_ECE", "0.06"))
    per_origin_toggle = os.getenv("MW_CAL_PER_ORIGIN", "false").lower() in ("1", "true", "yes")

    artifacts_dir = Path("artifacts")
    _ensure_dir(artifacts_dir)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=window_h)

    # load logs
    trig = _read_jsonl(models_dir / "trigger_history.jsonl")
    lab = _read_jsonl(models_dir / "label_feedback.jsonl")
    trig = [t for t in trig if (parse_ts(t.get("timestamp")) or now) >= cutoff]
    lab = [l for l in lab if (parse_ts(l.get("timestamp")) or now) >= cutoff]

    # join
    joined = _nearest_join_labels_to_triggers(lab, trig, join_min)

    # per-version collect (prob,label) and optional per-origin
    per_version_probs: DefaultDict[str, List[float]] = defaultdict(list)
    per_version_true: DefaultDict[str, List[int]] = defaultdict(list)
    per_version_origin: DefaultDict[str, DefaultDict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    per_version_origin_prob: DefaultDict[str, DefaultDict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for l, t in joined:
        if t is None:
            continue  # can't evaluate calibration without a matched score
        prob = _clip01(t.get("adjusted_score"))
        if prob is None:
            continue
        y = l.get("label")
        if y is True:
            yv = 1
        elif y is False:
            yv = 0
        else:
            continue

        ver = l.get("model_version") or t.get("model_version") or "unknown"
        org = (l.get("origin") or t.get("origin") or "unknown")

        per_version_probs[ver].append(prob)
        per_version_true[ver].append(yv)
        if per_origin_toggle:
            per_version_origin[ver][org].append(yv)
            per_version_origin_prob[ver][org].append(prob)

    # compute metrics
    per_version_out: List[Dict[str, Any]] = []
    for ver in sorted(per_version_probs.keys()):
        probs = per_version_probs[ver]
        trues = per_version_true[ver]
        n = len(probs)

        ece, bins_data = _ece_and_bins(probs, trues, n_bins)
        brier = _brier(probs, trues)
        alerts = _classify_alerts(ece, n, min_labels, max_ece)

        # optional per-origin worst ECEs
        worst_origins: List[Tuple[str, float, int]] = []
        if per_origin_toggle:
            for org, ys in per_version_origin[ver].items():
                ps = per_version_origin_prob[ver][org]
                e_org, _ = _ece_and_bins(ps, ys, n_bins)
                worst_origins.append((org, e_org, len(ps)))
            worst_origins.sort(key=lambda x: x[1], reverse=True)
            worst_origins = worst_origins[:3]

        # plot per version
        png_path = artifacts_dir / f"cal_reliability_{ver}.png"
        _plot_reliability(ver, bins_data, ece, n, png_path, ctx.is_demo)

        row: Dict[str, Any] = {
            "version": ver,
            "ece": round(float(ece), 6),
            "brier": round(float(brier), 6),
            "n": int(n),
            "bins": bins_data,
            "alerts": alerts,
            "demo": False,
        }
        if per_origin_toggle:
            row["worst_origins"] = [
                {"origin": o, "ece": round(e, 6), "n": int(nn)} for (o, e, nn) in worst_origins
            ]
        per_version_out.append(row)

    # demo seed if empty
    if not per_version_out and ctx.is_demo:
        # synthesize two plausible versions
        demo_versions = [
            ("v0.6.2", 0.04, 0.16, 120),
            ("v0.6.1", 0.09, 0.18, 80),
        ]
        for ver, ece_v, brier_v, n_v in demo_versions:
            # build a near-diagonal with slight bias
            xs = [i / n_bins + 0.5 / n_bins for i in range(n_bins)]
            ys = [min(1.0, max(0.0, x + (0.02 if ver.endswith("2") else -0.03))) for x in xs]
            bins_data = [{"p_hat": round(x, 4), "emp": round(y, 4), "n": max(1, n_v // n_bins)} for x, y in zip(xs, ys)]
            _plot_reliability(ver, bins_data, ece_v, n_v, artifacts_dir / f"cal_reliability_{ver}.png", True)
            per_version_out.append({
                "version": ver,
                "ece": ece_v,
                "brier": brier_v,
                "n": n_v,
                "bins": bins_data,
                "alerts": ["ok"] if ece_v <= max_ece and n_v >= min_labels else (["low_n"] if n_v < min_labels else ["high_ece"]),
                "demo": True,
            })

    # write JSON
    out_json = models_dir / "calibration_reliability.json"
    payload = {
        "window_hours": window_h,
        "bins": n_bins,
        "min_labels": min_labels,
        "max_ece": max_ece,
        "generated_at": _iso(now),
        "per_version": per_version_out,
        "demo": ctx.is_demo and (not per_version_out or any(r.get("demo") for r in per_version_out)),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # markdown
    tag = " (demo)" if payload.get("demo") else ""
    md.append(f"### 🧮 Calibration & Reliability ({window_h}h){tag}")
    if per_version_out:
        for r in per_version_out:
            alerts = ",".join(r.get("alerts", []))
            md.append(f"- `{r['version']}` → ECE={float(r['ece']):.3f} | Brier={float(r['brier']):.3f} | n={int(r['n'])} [{alerts}]")
    else:
        md.append("_no calibration data available_")