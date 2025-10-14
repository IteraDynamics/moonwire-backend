# scripts/summary_sections/volatility_aware_thresholds.py

from src.ml.infer import compute_volatility_adjusted_threshold
from scripts.summary_sections.common import pick_candidate_origins

def append(md, ctx, **kwargs):
    md.append("\n### 📉 Volatility-Aware Thresholds")
    try:
        cands = (ctx.candidates or pick_candidate_origins(ctx.origins_rows, ctx.yield_data, top=3))[:3]
        if not cands:
            md.append("_No candidate origins available._")
            return

        regimes_map = ctx.caches.get("volatility_regimes", {}) or {}
        dyn_map = ctx.caches.get("dyn_thresholds", {}) or {}

        for o in cands:
            base_thr = 0.5
            try:
                rec = dyn_map.get(o) or {}
                for key in ("used","dynamic","static"):
                    v = rec.get(key)
                    if v is not None:
                        base_thr = float(v); break
            except Exception:
                pass

            regime = regimes_map.get(o, "normal")
            try:
                res = compute_volatility_adjusted_threshold(float(base_thr), str(regime))
                if isinstance(res, dict):
                    adj_thr = float(res.get("adjusted_threshold", base_thr))
                    mult    = float(res.get("multiplier", 1.0))
                elif isinstance(res, (tuple, list)) and len(res) >= 2:
                    adj_thr, mult = float(res[0]), float(res[1])
                else:
                    adj_thr, mult = base_thr, 1.0
            except Exception:
                adj_thr, mult = base_thr, 1.0

            md.append(f"- {o}: Regime {regime} → multiplier={mult:.2f}")
            md.append(f"  - Threshold: base={base_thr:.3f} → adjusted={adj_thr:.3f}")
    except Exception as e:
        md.append(f"_⚠️ Volatility-aware section failed: {e}_")