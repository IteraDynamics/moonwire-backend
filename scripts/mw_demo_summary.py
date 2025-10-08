# scripts/mw_demo_summary.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Summary sections entrypoint
from scripts.summary_sections import build_all
from scripts.summary_sections.common import SummaryContext, ensure_dir, _iso


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


# --- Minimal seeding helpers so CI uploads never fail ---
_PNG_1x1_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
    b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
)

def _write_png_placeholder(path: Path, title_text: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa
        ensure_dir(path.parent)
        fig = plt.figure(figsize=(3, 2), dpi=100)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, title_text or "MoonWire", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(path))
        plt.close(fig)
        return
    except Exception:
        pass
    ensure_dir(path.parent)
    path.write_bytes(_PNG_1x1_BYTES)


def _seed_drift_response_plan(models_dir: Path) -> None:
    p = models_dir / "drift_response_plan.json"
    if p.exists(): return
    p.write_text(json.dumps({
        "generated_at": _iso(_now_utc()),
        "window_hours": 72,
        "candidates": [],
        "demo": True
    }, indent=2))


def _seed_retrain_plan(models_dir: Path) -> None:
    p = models_dir / "retrain_plan.json"
    if p.exists(): return
    p.write_text(json.dumps({
        "generated_at": _iso(_now_utc()),
        "action_mode": os.getenv("MW_RETRAIN_ACTION", "dryrun"),
        "candidates": [],
        "demo": True
    }, indent=2))


def _seed_ci_stub_artifacts(models_dir: Path, artifacts_dir: Path, logs_dir: Path) -> None:
    ensure_dir(models_dir); ensure_dir(artifacts_dir); ensure_dir(logs_dir)
    for name, payload in [
        ("calibration_per_origin.json", {"generated_at": _iso(_now_utc()), "origins": [], "demo": True}),
        ("calibration_reliability.json", {"generated_at": _iso(_now_utc()), "bins": 10, "ece": None, "curves": [], "demo": True}),
        ("model_registry.json", {"generated_at": _iso(_now_utc()), "models": [], "demo": True}),
    ]:
        p = models_dir / name
        if not p.exists():
            p.write_text(json.dumps(payload, indent=2))

    gl = logs_dir / "governance_actions.jsonl"
    if not gl.exists():
        gl.write_text(json.dumps({"ts": _iso(_now_utc()), "action": "demo_init"}) + "\n")

    # required plots for various upload globs
    _write_png_placeholder(artifacts_dir / "reddit_activity_demo.png", "reddit activity (demo)")
    _write_png_placeholder(artifacts_dir / "reddit_bursts_demo.png", "reddit bursts (demo)")
    _write_png_placeholder(artifacts_dir / "retrain_eval_demo.png", "retrain eval (demo)")
    _write_png_placeholder(artifacts_dir / "retrain_reliability_demo.png", "retrain reliability (demo)")
    _write_png_placeholder(artifacts_dir / "retrain_confusion_demo.png", "retrain confusion (demo)")
    _write_png_placeholder(artifacts_dir / "drift_response_timeline.png", "drift timeline (demo)")
    _write_png_placeholder(artifacts_dir / "drift_response_backtest_demo.png", "drift backtest (demo)")


@dataclass
class _Ctx(SummaryContext):
    logs_dir: Path
    models_dir: Path
    is_demo: bool
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    origins_rows: List[Dict[str, Any]] = field(default_factory=list)
    yield_data: Any = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    caches: Dict[str, Any] = field(default_factory=dict)


def _write_md(md_lines: List[str], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(md_lines))


def main() -> None:
    root = Path(".").resolve()
    models = root / "models"
    logs = root / "logs"
    arts = Path(os.getenv("ARTIFACTS_DIR", str(root / "artifacts")))
    ensure_dir(models); ensure_dir(logs); ensure_dir(arts)

    demo = str(os.getenv("DEMO_MODE", os.getenv("MW_DEMO", "false"))).lower() == "true"
    _seed_drift_response_plan(models)
    _seed_retrain_plan(models)
    _seed_ci_stub_artifacts(models, arts, logs)

    ctx = _Ctx(logs_dir=logs, models_dir=models, is_demo=demo, artifacts_dir=arts)
    md_lines = build_all(ctx)

    header = ["MoonWire CI Demo Summary"]
    all_lines = header + md_lines + ["Job summary generated at run-time"]
    _write_md(all_lines, arts / "demo_summary.md")


if __name__ == "__main__":
    main()
