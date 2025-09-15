# scripts/summary_sections/score_distribution.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone, timedelta
from statistics import median
import json, random

import matplotlib
matplotlib.use("Agg")  # headless for CI
import matplotlib.pyplot as plt

from .common import SummaryContext, is_demo_mode, parse_ts


def _load_jsonl_safe(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    try:
        for ln in p.read_text(encoding="utf-8").splitlines():
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


def _p90(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    # simple, robust p90 for small n
    idx = max(0, min(len(s) - 1, int(round(0.9 * (len(s) - 1)))))
    return float(s[idx])


def _fmt(v: float, nd: int = 3) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "n/a"


def _is_drifted(row: dict) -> bool:
    """
    Heuristic for 'drifted' — treat as drifted if:
      - 'drifted_features' exists and is non-empty, or
      - 'drifted' or 'drift' is True-ish
    """
    try:
        if isinstance(row.get("drifted_features"), list) and row["drifted_features"]:
            return True
    except Exception:
        pass
    for k in ("drifted", "drift"):
        v = row.get(k)
        if isinstance(v, bool) and v:
            return True
        # allow truthy strings like "true"
        if isinstance(v, str) and v.strip().lower() in ("1", "true", "yes"):
            return True
    return False


def _score_of(row: dict) -> float | None:
    """
    Prefer adjusted_score; fall back to prob_trigger_next_6h or ensemble_score.
    """
    for k in ("adjusted_score", "prob_trigger_next_6h", "ensemble_score"):
        v = row.get(k)
        if isinstance(v, (int, float)):
            try:
                f = float(v)
                if 0.0 <= f <= 1.0:
                    return f
                # tolerate logit-like scores by squashing (rare)
                if f > 1.0:
                    return max(0.0, min(1.0, f))
            except Exception:
                continue
    return None


def append(md: list[str], ctx: SummaryContext, hours: int = 48, min_points: int = 12) -> None:
    """
    Renders '📐 Score Distribution (48h)' with an overlay of drifted vs non-drifted scores
    and saves a stacked histogram to artifacts/score_hist_drift_overlay_48h.png.

    Also keeps the earlier summary stats line for continuity.
    """
    md.append("\n### 📐 Score Distribution (48h)")

    # ---- load recent trigger history ----
    hist_path = ctx.models_dir / "trigger_history.jsonl"
    rows = _load_jsonl_safe(hist_path)

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    drifted, clean = [], []
    for r in rows:
        ts = parse_ts(r.get("timestamp"))
        if not ts or ts < cutoff:
            continue
         # pick a score
        s = _score_of(r)
        if s is None:
            continue
        if _is_drifted(r):
            drifted.append(s)
        else:
            clean.append(s)

    seeded = False
    total = len(drifted) + len(clean)

    # ---- demo seeding if not enough data ----
    if total < min_points and is_demo_mode():
        # Create a plausible mix: non-drifted slightly higher scores; drifted slightly lower
        rng = random.Random(42)
        # target totals: roughly 64 with a 3:1 clean:drifted mix
        n_clean = max(8, int(0.75 * max(min_points, 64)))
        n_drift = max(4, int(0.25 * max(min_points, 64)))
        clean = [max(0.0, min(1.0, rng.betavariate(2.5, 6.0))) for _ in range(n_clean)]
        drifted = [max(0.0, min(1.0, rng.betavariate(2.0, 9.0))) for _ in range(n_drift)]
        seeded = True
        total = len(drifted) + len(clean)

    if total == 0:
        md.append("_No recent scores in the last 48h._")
        return

    # ---- summary stats (overall, like before) ----
    all_scores = clean + drifted
    mean_val = sum(all_scores) / float(len(all_scores))
    med_val = median(all_scores)
    p90_val = _p90(all_scores)

    if seeded:
        md.append("_(demo) synthesized scores for visibility_")
    md.append(f"- n={total}, mean={_fmt(mean_val)}, median={_fmt(med_val)}, p90={_fmt(p90_val)}")

    # ---- drift split counts ----
    md.append(f"- split: drifted={len(drifted)} | non-drifted={len(clean)}")

    # ---- render histogram (stacked overlay) ----
    ART = Path("artifacts"); ART.mkdir(exist_ok=True)
    overlay_path = ART / "score_hist_drift_overlay_48h.png"

    try:
        # common bins across both groups: 0.00..1.00 in 0.02 steps (51 edges)
        bins = [i / 50.0 for i in range(51)]
        plt.figure(figsize=(6.0, 2.4))
        # Stacked so the total per bin is immediately visible; legend differentiates groups
        plt.hist([clean, drifted],
                 bins=bins,
                 stacked=True,
                 label=["non-drifted", "drifted"],
                 alpha=0.85)
        plt.title("Score distribution (48h)")
        plt.xlabel("score")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(overlay_path)
        plt.close()

        # Inline the image in the summary
        md.append(f"  \n![]({overlay_path.as_posix()})")
    except Exception as e:
        md.append(f"_⚠️ Histogram render failed: {type(e).__name__}: {e}_")