from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Any, Dict

from scripts.governance.drift_response import Paths, run_policy


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "y", "on")


def append(md: List[str], ctx: Any) -> None:
    """
    Summary section for Automated Drift Response.
    Produces plan JSON, plots, and renders a concise markdown table.
    """
    # Support simple test contexts (may not have artifacts_dir)
    models_dir: Path = Path(getattr(ctx, "models_dir", "models"))
    logs_dir: Path = Path(getattr(ctx, "logs_dir", "logs"))
    artifacts_dir: Path = Path(getattr(ctx, "artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        plan_path = run_policy(Paths(models_dir=models_dir, logs_dir=logs_dir, artifacts_dir=artifacts_dir))
        plan = json.loads(plan_path.read_text())
    except Exception as e:
        md.append(f"\n> ⚠️ Drift Response failed: `{type(e).__name__}: {e}`\n")
        return

    mode = (plan.get("action_mode") or "dryrun").lower()
    demo = plan.get("demo", False)

    md.append(f"### 🛡️ Automated Drift Response ({plan.get('window_hours', 72)}h) — mode: {mode}{' (demo)' if demo else ''}")

    rows = plan.get("candidates", []) or []
    if not rows:
        md.append("_no candidates detected_")
        return

    for r in rows:
        origin = r.get("origin", "unknown")
        ver = r.get("model_version", "v0")
        dth = r.get("delta", 0.0)
        bt = r.get("backtest", {})
        prec = bt.get("precision_delta", 0.0)
        ece = bt.get("ece_delta", 0.0)
        rec = bt.get("recall_delta", 0.0)
        decision = r.get("decision", "hold")
        reasons = r.get("reasons", [])
        line = (
            f"{origin}/{ver}  → {dth:+.2f} threshold  "
            f"| ΔPrecision {prec:+.1%} | ΔECE {ece:+.3f} | ΔRecall {rec:+.1%} "
            f"[{decision}]"
        )
        if reasons:
            line += f"  — reason: {', '.join(reasons)}"
        md.append(line)

    md.append("_Guardrails: min_precision ≥0.75, min_labels ≥10, max_shift ±0.10; cooldown 12h._")
