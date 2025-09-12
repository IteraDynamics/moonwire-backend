# scripts/summary_sections/accuracy_by_version_section.py
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import List, Dict, Any


def _parse_semver(v: str) -> tuple[int, int, int]:
    s = str(v or "")
    if s.startswith("v"):
        s = s[1:]
    s = s.split("-", 1)[0]  # drop suffix like "-demo"
    parts = s.split(".")
    nums: list[int] = []
    for i in range(3):
        try:
            nums.append(int(parts[i]))
        except Exception:
            nums.append(-1)
    return tuple(nums)  # (major, minor, patch)


def render(md: List[str], models_dir: Path | str = "models") -> None:
    """
    Render the '🧪 Accuracy by Model Version' section into `md`.

    Args:
        md: markdown list to append to (in-place).
        models_dir: models directory (Path or str), default "models".
    """
    md.append("\n### 🧪 Accuracy by Model Version")

    try:
        # Lazy imports so this module is safe in environments without full ML stack
        from src.ml.metrics import compute_accuracy_by_version
    except Exception as e:
        md.append(f"_⚠️ Accuracy by version unavailable: {type(e).__name__}: {e}_")
        return

    mdir = Path(models_dir)
    trig_path = mdir / "trigger_history.jsonl"
    lab_path = mdir / "label_feedback.jsonl"

    # window via env, default 72h
    try:
        window_h = int(os.getenv("MW_ACCURACY_WINDOW_H", "72"))
    except Exception:
        window_h = 72

    # Compute
    try:
        res: Dict[str, Dict[str, Any]] | None = compute_accuracy_by_version(
            trig_path, lab_path, window_hours=window_h
        )
    except Exception as e:
        md.append(f"_⚠️ Accuracy by version failed to compute: {e}_")
        return

    # Persist snapshot for diffs across runs (best-effort)
    try:
        snap_path = mdir / "accuracy_by_version.json"
        with snap_path.open("w", encoding="utf-8") as f:
            json.dump(res or {}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Nothing yet? Provide demo or waiting message.
    if not res or all(str(k).startswith("_") for k in res.keys()):
        demo_mode = os.getenv("DEMO_MODE", "false").lower() in ("1", "true", "yes")
        if demo_mode:
            demo_rows = [
                ("v0.5.2", {"precision": 0.67, "recall": 0.50, "f1": 0.57, "tp": 2, "fp": 1, "fn": 2, "n": 5}),
                ("v0.5.1", {"precision": 1.00, "recall": 0.33, "f1": 0.50, "tp": 1, "fp": 0, "fn": 2, "n": 3}),
            ]
            for ver, m in demo_rows:
                md.append(
                    f"- {ver} → precision={m['precision']:.2f}, recall={m['recall']:.2f}, "
                    f"F1={m['f1']:.2f} (tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, n={m['n']})"
                )
        else:
            md.append("_Waiting for more feedback..._")
        return

    # Show versions (exclude meta keys that start with "_")
    items = [(ver, m) for ver, m in res.items() if not str(ver).startswith("_")]
    # sort by sample size desc, then semver desc
    items.sort(key=lambda kv: (kv[1].get("n", 0), _parse_semver(kv[0])), reverse=True)

    for ver, m in items:
        suffix = " (low n)" if m.get("n", 0) < 5 else ""
        md.append(
            f"- {ver} → precision={m['precision']:.2f}, recall={m['recall']:.2f}, "
            f"F1={m['f1']:.2f} (tp={m['tp']}, fp={m['fp']}, fn={m['fn']}, n={m['n']}){suffix}"
        )

    # micro & macro lines (optional keys computed by metrics helper)
    micro = res.get("_micro")
    macro = res.get("_macro")
    if micro:
        md.append(
            f"- Overall (micro) → precision={micro['precision']:.2f}, recall={micro['recall']:.2f}, "
            f"F1={micro['f1']:.2f} (tp={micro['tp']}, fp={micro['fp']}, fn={micro['fn']}, n={micro['n']})"
        )
    if macro:
        md.append(
            f"- Macro avg → precision={macro['precision']:.2f}, recall={macro['recall']:.2f}, "
            f"F1={macro['f1']:.2f} (versions={macro['versions']})"
        )