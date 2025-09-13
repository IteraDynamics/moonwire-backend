# scripts/summary_sections/trigger_likelihood_v0.py

from datetime import datetime, timezone
from scripts.summary_sections.common import pick_candidate_origins
from src.ml.infer import score as infer_score, model_metadata
from .common import _build_summary_features_for_origin

def append(md, ctx, **kwargs):
    md.append("\n### 🤖 Trigger Likelihood v0 (next 6h)")
    # metadata line (created_at, AUC, demo flag)
    try:
        _meta = model_metadata() or {}
        bits = []
        if _meta.get("created_at"): bits.append(f"model@{_meta['created_at']}")
        auc = (_meta.get("metrics") or {}).get("roc_auc_va") or (_meta.get("metrics") or {}).get("roc_auc_tr")
        if auc is not None:
            try: bits.append(f"AUC={float(auc):.2f}")
            except Exception: bits.append(f"AUC={auc}")
        if _meta.get("demo"): bits.append("demo")
        if bits: md.append("- " + " • ".join(bits))
    except Exception:
        pass

    try:
        cands = ctx.candidates or pick_candidate_origins(ctx.origins_rows, ctx.yield_data, top=3)
        ctx.candidates = list(cands)  # persist for later
        now_bucket = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0).isoformat()

        printed = 0
        for o in cands:
            try:
                res = infer_score({"origin": o, "timestamp": now_bucket})
                p = res.get("prob_trigger_next_6h")
                if isinstance(p, (int, float)):
                    line = f"- {o}: **{p*100:.1f}%** chance of trigger in next 6h"
                    contribs = res.get("contributions") or {}
                    if contribs:
                        top = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
                        line += " (" + ", ".join(f"{k}={v:+.2f}" for k, v in top) + ")"
                    md.append(line); printed += 1
            except Exception:
                continue
        if printed == 0:
            md.append("- _No score available._")
    except Exception as e:
        md.append(f"_⚠️ Trigger score failed: {e}_")