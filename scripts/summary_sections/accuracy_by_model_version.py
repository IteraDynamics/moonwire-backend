# scripts/summary_sections/accuracy_by_model_version.py
from __future__ import annotations
import os, json
from scripts.summary_sections.common import SummaryContext

def _parse_semver(v: str):
    v = str(v)
    if v.startswith("v"): v = v[1:]
    parts = v.split("-", 1)[0].split(".")
    nums = []
    for i in range(3):
        try: nums.append(int(parts[i]))
        except Exception: nums.append(-1)
    return tuple(nums)

def append(md: list[str], ctx: SummaryContext):
    md.append("\n### 🧪 Accuracy by Model Version")
    try:
        from src.ml.metrics import compute_accuracy_by_version
    except Exception as e:
        md.append(f"_unavailable: {type(e).__name__}_")
        return

    try:
        window_h = int(os.getenv("MW_ACCURACY_WINDOW_H", "72"))
    except Exception:
        window_h = 72

    trig_path = ctx.models_dir / "trigger_history.jsonl"
    lab_path  = ctx.models_dir / "label_feedback.jsonl"

    try:
        res = compute_accuracy_by_version(trig_path, lab_path, window_hours=window_h) or {}
    except Exception as e:
        md.append(f"_compute failed: {type(e).__name__}_")
        return

    items = [(ver, m) for ver, m in res.items() if not str(ver).startswith("_")]
    if not items:
        if os.getenv("DEMO_MODE","false").lower() in ("1","true","yes"):
            demo_rows = [
                ("v0.5.2", {"precision": 0.67, "recall": 0.50, "f1": 0.57, "tp": 2, "fp": 1, "fn": 2, "n": 5}),
                ("v0.5.1", {"precision": 1.00, "recall": 0.33, "f1": 0.50, "tp": 1, "fp": 0, "fn": 2, "n": 3}),
            ]
            for ver, m in demo_rows:
                md.append(f"- {ver} → precision={m['precision']:.2f}, recall={m['recall']:.2f}, "
                          f"F1={m['f1']:.2f} (tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, n={m['n']})")
        else:
            md.append("_Waiting for more feedback..._")
        return

    items.sort(key=lambda kv: (kv[1].get("n", 0), _parse_semver(kv[0])), reverse=True)
    for ver, m in items:
        suffix = " (low n)" if m.get("n", 0) < 5 else ""
        md.append(f"- {ver} → precision={m['precision']:.2f}, recall={m['recall']:.2f}, "
                  f"F1={m['f1']:.2f} (tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, n={m['n']}){suffix}")

    micro = res.get("_micro"); macro = res.get("_macro")
    if micro:
        md.append(f"- Overall (micro) → precision={micro['precision']:.2f}, recall={micro['recall']:.2f}, "
                  f"F1={micro['f1']:.2f} (tp={micro['tp']}, fp={micro['fp']}, fn={micro['fn']}, n={micro['n']})")
    if macro:
        md.append(f"- Macro avg → precision={macro['precision']:.2f}, recall={macro['recall']:.2f}, "
                  f"F1={macro['f1']:.2f} (versions={macro['versions']})")