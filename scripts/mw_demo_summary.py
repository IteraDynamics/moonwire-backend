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

# --------------------------
# Demo data seed (kept stable for tests)
# --------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)

def generate_demo_data_if_needed(reviewers: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    - Non-demo: pass-through (reviewers, []).
    - Demo + reviewers provided: pass-through (reviewers, []).
    - Demo + reviewers empty: synthesize 3 reviewers AND emit one event per reviewer.
    """
    demo = str(os.getenv("DEMO_MODE", os.getenv("MW_DEMO", "false"))).lower() == "true"
    if not demo:
        return reviewers, []
    if reviewers:
        return reviewers, []

    now = _now_utc()
    out_reviewers: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    seeds = [
        {"id": "rev_demo_1", "origin": "reddit", "score": 0.82},
        {"id": "rev_demo_2", "origin": "rss_news", "score": 0.54},
        {"id": "rev_demo_3", "origin": "twitter", "score": 0.67},
    ]
    for i, r in enumerate(seeds):
        rcopy = dict(r)
        rcopy["timestamp"] = _iso(now - timedelta(hours=max(0, 2 - i)))
        out_reviewers.append(rcopy)
        events.append(
            {
                "type": "demo_review_created",
                "review_id": rcopy["id"],
                "at": _iso(now - timedelta(hours=max(0, 2 - i))),
                "meta": {"note": "seeded in demo mode", "version": "v0.6.6"},
            }
        )
    return out_reviewers, events

# --------------------------
# Seed governance demo artifacts when missing
# --------------------------

def _seed_drift_response_plan(models_dir: Path) -> None:
    ensure_dir(models_dir)
    jpath = models_dir / "drift_response_plan.json"
    if jpath.exists():
        return
    now = _now_utc()
    plan = {
        "generated_at": _iso(now),
        "window_hours": 72,
        "grace_hours": int(os.getenv("MW_DRIFT_GRACE_H", "6")),
        "min_buckets": int(os.getenv("MW_DRIFT_MIN_BUCKETS", "3")),
        "ece_threshold": float(os.getenv("MW_DRIFT_ECE_THRESH", "0.06")),
        "action_mode": os.getenv("MW_DRIFT_ACTION", "dryrun"),
        "candidates": [],
        "demo": True,
    }
    jpath.write_text(json.dumps(plan))

def _seed_retrain_plan(models_dir: Path) -> None:
    ensure_dir(models_dir)
    jpath = models_dir / "retrain_plan.json"
    if jpath.exists():
        return
    now = _now_utc()
    plan = {
        "generated_at": _iso(now),
        "action_mode": os.getenv("MW_RETRAIN_ACTION", "dryrun"),
        "candidates": [],
        "demo": True,
    }
    jpath.write_text(json.dumps(plan))

# --------------------------
# CI stub artifacts (demo-friendly)
# --------------------------

_PNG_1x1_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
    b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
)

def _write_png_placeholder(path: Path, title_text: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa: E402
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

def _seed_ci_stub_artifacts(models_dir: Path, artifacts_dir: Path, logs_dir: Path) -> None:
    ensure_dir(models_dir); ensure_dir(artifacts_dir); ensure_dir(logs_dir)

    cal_per_origin = models_dir / "calibration_per_origin.json"
    if not cal_per_origin.exists():
        cal_per_origin.write_text(json.dumps({
            "generated_at": _iso(_now_utc()),
            "window_hours": int(os.getenv("MW_CAL_WINDOW_H", "72")),
            "origins": [],
            "demo": True
        }, indent=2))

    cal_reliability = models_dir / "calibration_reliability.json"
    if not cal_reliability.exists():
        cal_reliability.write_text(json.dumps({
            "generated_at": _iso(_now_utc()),
            "bins": int(os.getenv("MW_CAL_BINS", "10")),
            "ece": None,
            "curves": [],
            "demo": True
        }, indent=2))

    model_registry = models_dir / "model_registry.json"
    if not model_registry.exists():
        model_registry.write_text(json.dumps({
            "generated_at": _iso(_now_utc()),
            "models": [],
            "demo": True
        }, indent=2))

    gov_log = logs_dir / "governance_actions.jsonl"
    if not gov_log.exists():
        gov_log.write_text(json.dumps({
            "ts": _iso(_now_utc()),
            "action": "demo_init",
            "meta": {"note": "seeded for CI uploads"}
        }) + "\n")

    _write_png_placeholder(artifacts_dir / "reddit_activity_demo.png", "reddit activity (demo)")
    _write_png_placeholder(artifacts_dir / "reddit_bursts_demo.png", "reddit bursts (demo)")
    _write_png_placeholder(artifacts_dir / "retrain_eval_demo.png", "retrain eval (demo)")
    _write_png_placeholder(artifacts_dir / "retrain_reliability_demo.png", "retrain reliability (demo)")
    _write_png_placeholder(artifacts_dir / "retrain_confusion_demo.png", "retrain confusion (demo)")
    _write_png_placeholder(artifacts_dir / "drift_response_timeline.png", "drift timeline (demo)")
    _write_png_placeholder(artifacts_dir / "drift_response_backtest_demo.png", "drift backtest (demo)")

def _seed_versioned_model_stub(models_dir: Path, version: str = "v0.5.1") -> None:
    vdir = ensure_dir(models_dir / version)
    has_real = any((vdir / name).exists() for name in (
        "model.joblib", "model.meta.json", "README.txt", "README.md"
    ))
    if has_real:
        return
    readme = vdir / "README.txt"
    meta = vdir / "stub.meta.json"
    now = _now_utc()
    if not readme.exists():
        readme.write_text(
            "MoonWire demo stub for versioned artifacts.\n"
            f"Generated at { _iso(now) } (demo mode).\n"
        )
    if not meta.exists():
        meta.write_text(json.dumps({
            "generated_at": _iso(now),
            "version": version,
            "kind": "demo_stub",
            "note": "Created so CI artifact upload models/v*/** has a match during demos."
        }, indent=2))

# --------------------------
# Context
# --------------------------

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

# --------------------------
# Markdown assembly / normalization
# --------------------------

HEADER_PLAIN = "MoonWire CI Demo Summary"
HEADER_DETAILED_PREFIX = "MoonWire Demo Summary —"
FOOTER_LINE = "Job summary generated at run-time"

# Odd separators & invisibles that show up in CI logs
_SEP_CHARS = [
    "\u2028",  # LINE SEPARATOR
    "\u2029",  # PARAGRAPH SEPARATOR
    "\u0085",  # NEXT LINE
    "\r\n", "\r", "\n",
]
_INVISIBLE_WS = {
    "\u00A0": " ",  # NBSP
    "\u202F": " ",  # NNBSP
    "\u2009": " ",  # THIN SPACE
    "\u200A": " ",  # HAIR SPACE
    "\u200B": "",   # ZERO WIDTH SPACE
    "\u2060": "",   # WORD JOINER
    "\ufeff": "",   # BOM
}

def _canon_text(s: str) -> str:
    # Replace invisibles, collapse spaces, strip
    for k, v in _INVISIBLE_WS.items():
        s = s.replace(k, v)
    s = " ".join(s.split())
    return s.strip()

def _explode_weird_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for ln in lines:
        pieces = [ln]
        for sep in _SEP_CHARS:
            tmp: List[str] = []
            for p in pieces:
                tmp.extend(p.split(sep))
            pieces = tmp
        out.extend(p for p in pieces if p is not None)
    return [p for p in (x.strip() for x in out) if p is not None]

def _is_plain_header(s: str) -> bool:
    return _canon_text(s).casefold() == HEADER_PLAIN.casefold()

def _is_detailed_header(s: str) -> bool:
    return _canon_text(s).startswith(HEADER_DETAILED_PREFIX)

def _is_footer(s: str) -> bool:
    return _canon_text(s).casefold() == FOOTER_LINE.casefold()

def _normalize_whole_markdown(lines: List[str]) -> List[str]:
    """
    Global cleanup:
      • Prefer a single timestamped header if present; otherwise one plain header.
      • Drop all extra plain headers anywhere in the body (even with invisibles).
      • Keep only one footer, placed at the very end.
      • Collapse consecutive blank lines.
      • Robust to Unicode separators & zero-width/invisible spaces.
    """
    exploded = _explode_weird_lines(lines)

    demo_header_line: str | None = None
    body: List[str] = []
    saw_any_plain_header = False

    for raw in exploded:
        s = raw.strip()
        if not s:
            body.append("")
            continue
        if _is_detailed_header(s) and demo_header_line is None:
            demo_header_line = _canon_text(s)
            continue
        if _is_plain_header(s):
            saw_any_plain_header = True
            continue
        if _is_footer(s):
            continue
        body.append(s)

    out: List[str] = []
    if demo_header_line is not None:
        out.append(HEADER_PLAIN)
        out.append(demo_header_line)
    elif saw_any_plain_header:
        out.append(HEADER_PLAIN)
    else:
        out.append(HEADER_PLAIN)

    # Collapse consecutive blanks
    compact: List[str] = []
    prev_blank = False
    for s in body:
        cs = _canon_text(s)
        is_blank = (cs == "")
        if is_blank and prev_blank:
            continue
        compact.append(cs if cs != "" else "")
        prev_blank = is_blank

    # Remove any stray plain headers that slipped into body
    compact = [ln for ln in compact if not _is_plain_header(ln)]

    out.extend(compact)

    if not out or not _is_footer(out[-1]):
        out.append(FOOTER_LINE)
    return out

def _write_md(md_lines: List[str], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(md_lines))

# --------------------------
# Main
# --------------------------

def main() -> None:
    # workspace paths
    root = Path(".").resolve()
    models = root / "models"
    logs = root / "logs"
    arts = Path(os.getenv("ARTIFACTS_DIR", str(root / "artifacts")))
    ensure_dir(models); ensure_dir(logs); ensure_dir(arts)

    # ensure governance artifacts exist for CI rendering
    demo = str(os.getenv("DEMO_MODE", os.getenv("MW_DEMO", "false"))).lower() == "true"
    _seed_drift_response_plan(models)   # safe in either mode
    _seed_retrain_plan(models)          # safe in either mode

    # stub artifacts so upload globs always match
    _seed_ci_stub_artifacts(models, arts, logs)

    # ensure versioned bundle exists in demo to satisfy models/v0.5.1/** upload
    if demo:
        _seed_versioned_model_stub(models, version=os.getenv("MODEL_VERSION", "v0.5.1"))

    # assemble markdown from sections
    ctx = _Ctx(logs_dir=logs, models_dir=models, is_demo=demo, artifacts_dir=arts)
    section_lines = build_all(ctx)

    # Wrap with simple header/footer and then normalize globally
    raw_lines: List[str] = [HEADER_PLAIN] + section_lines + [FOOTER_LINE]
    final_lines = _normalize_whole_markdown(raw_lines)

    # write to artifacts
    _write_md(final_lines, arts / "demo_summary.md")

if __name__ == "__main__":
    main()