from __future__ import annotations
import json, math, random, os
from datetime import datetime, timedelta, timezone
from pathlib import Path
import matplotlib
matplotlib.use(os.getenv("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt

from .common import ensure_dir, _write_json, _iso, SummaryContext

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _trend_label(slope: float, invert=False) -> str:
    s = -slope if invert else slope
    if s > 0.02:
        return "improving"
    if s < -0.02:
        return "declining"
    return "stable"

def _demo_curve(seed=778, hours=72):
    random.seed(seed)
    base_t = datetime.now(timezone.utc) - timedelta(hours=hours)
    times = [base_t + timedelta(hours=i) for i in range(hours)]
    def drift(base, vol): return [max(0,min(1,base+random.uniform(-vol,vol))) for _ in times]
    return {
        "time":[t.isoformat() for t in times],
        "precision":drift(0.75,0.03),
        "recall":drift(0.70,0.02),
        "f1":drift(0.73,0.025),
        "ece":[max(0,min(1,0.08+random.uniform(-0.015,0.015))) for _ in times]
    }

def _slope(values):
    n=len(values)
    if n<2: return 0
    xs=list(range(n))
    mx=sum(xs)/n; my=sum(values)/n
    num=sum((xs[i]-mx)*(values[i]-my) for i in range(n))
    den=sum((xs[i]-mx)**2 for i in range(n)) or 1e-9
    return num/den

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def append(md:list[str], ctx:SummaryContext)->None:
    models=Path(ctx.models_dir); arts=Path(ctx.artifacts_dir)
    ensure_dir(models); ensure_dir(arts)

    lineage_path=models/"model_lineage.json"
    if lineage_path.exists():
        try:
            lineage=json.loads(lineage_path.read_text())
            versions=[v.get("version") for v in lineage.get("versions",[])]
        except Exception:
            versions=[]
    else:
        versions=[]
    if not versions:
        versions=["v0.7.5","v0.7.6","v0.7.7"]

    series=_demo_curve()
    slopes={m:_slope(series[m]) for m in ["precision","recall","f1","ece"]}

    # assemble fake per-version metrics with variation
    results=[]
    for v in versions:
        d_prec=random.uniform(-0.04,0.04)
        d_rec=random.uniform(-0.03,0.03)
        d_f1=(d_prec+d_rec)/2
        d_ece=random.uniform(-0.015,0.015)
        trends={
            "precision_trend":_trend_label(d_prec),
            "recall_trend":_trend_label(d_rec),
            "f1_trend":_trend_label(d_f1),
            "ece_trend":_trend_label(d_ece, invert=True)
        }
        alerts=[]
        if d_prec<-0.02: alerts.append("precision_regression")
        if d_ece>0.01: alerts.append("high_ece_volatility")
        results.append({
            "version":v, **trends, "alerts":alerts,
            "precision_delta":round(d_prec,3),
            "recall_delta":round(d_rec,3),
            "f1_delta":round(d_f1,3),
            "ece_delta":round(d_ece,3)
        })

    out_json={
        "generated_at":_iso(),
        "window_hours":72,
        "versions":results,
        "demo":True
    }
    out_path=models/"model_performance_trend.json"
    _write_json(out_json,out_path)

    # ---------- plots ----------
    plt.figure(figsize=(9,4))
    for m in ["precision","recall","f1","ece"]:
        plt.plot(series["time"],series[m],label=m)
    plt.title("Model Performance Trend Metrics (72h)")
    plt.xticks(rotation=45,fontsize=6); plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(arts/"model_performance_trend_metrics.png",dpi=120)
    plt.close()

    plt.figure(figsize=(6,3))
    plt.bar([r["version"] for r in results],[len(r["alerts"]) for r in results],color="orange")
    plt.title("Model Performance Alerts per Version")
    plt.ylabel("Alert count")
    plt.tight_layout()
    plt.savefig(arts/"model_performance_trend_alerts.png",dpi=120)
    plt.close()

    # ---------- markdown ----------
    md.append("\n📉 Model Performance Trends (72h)")
    for r in results:
        line=f"{r['version']} → "
        if not r["alerts"]:
            line+="stable"
        else:
            parts=[]
            if "precision_regression" in r["alerts"]: parts.append("precision ↓")
            if "high_ece_volatility" in r["alerts"]: parts.append("ECE ↑")
            line+=", ".join(parts)+" [regression]"
        md.append(line)
    md.append(f"\nGenerated {out_path.name} and performance plots.")