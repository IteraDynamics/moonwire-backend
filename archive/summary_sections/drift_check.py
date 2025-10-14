# scripts/summary_sections/drift_check.py
from __future__ import annotations
from scripts.summary_sections.common import SummaryContext

_DRIFT_SCORE_MIN = 0.6

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### 🔎 Drift Check (features)")
    # Expect a precomputed drift object placed by upstream jobs; be defensive.
    drift_raw = (
        ctx.caches.get("drift")
        or ctx.caches.get("drift_result")
        or ctx.caches.get("drift_check")
        or {}
    )
    items = drift_raw.get("features") or drift_raw.get("items") or []

    norm_items = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            feat = it.get("feature") or it.get("name") or "feature"
            score = float(it.get("score", it.get("drift_score", 0.0) or 0.0))
            dmean = float(it.get("delta_mean", it.get("delta", 0.0) or 0.0))
            nz_tr = float(
                it.get("nz_train")
                or it.get("nz_pct_train")
                or it.get("nz_train_pct")
                or it.get("nz_train_percent")
                or it.get("train_nonzero_pct")
                or 0.0
            )
            nz_lv = float(
                it.get("nz_live")
                or it.get("nz_pct_live")
                or it.get("nz_live_pct")
                or it.get("nz_live_percent")
                or it.get("live_nonzero_pct")
                or 0.0
            )
        except Exception:
            continue
        norm_items.append(
            {"feature": feat, "score": score, "delta_mean": dmean, "nz_train": nz_tr, "nz_live": nz_lv}
        )

    top = [x for x in norm_items if x["score"] >= _DRIFT_SCORE_MIN]
    top.sort(key=lambda x: x["score"], reverse=True)
    top = top[:3]

    if not top:
        md.append("No material drift detected.")
    else:
        for x in top:
            md.append(
                f"- {x['feature']}: Δmean={round(x['delta_mean'], 2)}, "
                f"nz% {round(x['nz_train'])}→{round(x['nz_live'])}, "
                f"score={round(x['score'], 2)}"
            )