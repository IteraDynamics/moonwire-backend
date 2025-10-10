"""
Governance Apply & Safeguards (v0.8.0)

Turns recommendations in models/model_governance_actions.json into optional,
logged, and reversible actions. Safe-by-default (dryrun), guardrails enforced,
append-only ledger, and a simple timeline PNG artifact for CI.

Public API: append(md, ctx)
"""

from __future__ import annotations
import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

# --- small, local helpers (no external imports to avoid regressions) ---

def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name)
    return v if v is not None else (default or "")

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1","true","t","yes","y","on")

def _iso(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now(timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00","Z")

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        if p.exists():
            return json.loads(p.read_text())
    except Exception:
        return None
    return None

def _append_jsonl(p: Path, row: Dict[str, Any]) -> None:
    _ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()

def _hours_ago(ts_seconds: float) -> float:
    return ( _now_ts() - ts_seconds ) / 3600.0

def _load_jsonl_latest(p: Path, limit: int = 256) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not p.exists():
        return rows
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return []
    return rows[-limit:]

# --- policy / thresholds ---

@dataclass
class Guardrails:
    min_labels: int = 20
    min_precision: float = 0.75
    max_ece: float = 0.06
    cooldown_h: float = 24.0

def _guardrails_from_env() -> Guardrails:
    return Guardrails(
        min_labels = int(_env("MW_GOV_MIN_LABELS", "20") or "20"),
        min_precision = float(_env("MW_GOV_MIN_PRECISION", "0.75") or "0.75"),
        max_ece = float(_env("MW_GOV_MAX_ECE", "0.06") or "0.06"),
        cooldown_h = float(_env("MW_GOV_COOLDOWN_H", "24") or "24"),
    )

# --- metrics extraction from trend (best-effort; resilient) ---

def _metrics_for_version(trend: Dict[str, Any], version: str) -> Dict[str, Any]:
    for v in trend.get("versions", []):
        if str(v.get("version")) == str(version):
            return {
                "labels": int(v.get("labels", 0) or 0),
                "precision": float(v.get("precision", v.get("precision_est", 0.0) or 0.0)),
                "ece": float(v.get("ece", v.get("ece_est", 0.0) or 0.0)),
                "precision_delta": float(v.get("precision_delta", 0.0) or 0.0),
                "ece_delta": float(v.get("ece_delta", 0.0) or 0.0),
                "precision_trend": str(v.get("precision_trend","") or ""),
                "ece_trend": str(v.get("ece_trend","") or ""),
            }
    return {
        "labels": 0, "precision": 0.0, "ece": 1.0,
        "precision_delta": 0.0, "ece_delta": 0.0,
        "precision_trend": "", "ece_trend": ""
    }

# --- cooldown logic ---

def _last_action_hours(rows: List[Dict[str, Any]], version: str) -> Optional[float]:
    last_ts: Optional[float] = None
    for r in rows:
        if str(r.get("version")) == str(version):
            ts = r.get("ts")
            if isinstance(ts, (int, float)):
                last_ts = ts if (last_ts is None or ts > last_ts) else last_ts
    if last_ts is None:
        return None
    return _hours_ago(last_ts)

# --- lineage helpers for promote/rollback ---

def _lineage_versions(lineage: Dict[str, Any]) -> List[str]:
    return [str(v.get("version")) for v in lineage.get("versions", []) if v.get("version")]

def _current_model(models_dir: Path) -> Optional[str]:
    f = models_dir / "current_model.txt"
    if not f.exists():
        return None
    try:
        return f.read_text().strip()
    except Exception:
        return None

def _write_current_model(models_dir: Path, version: str) -> None:
    f = models_dir / "current_model.txt"
    _ensure_dir(models_dir)
    f.write_text(str(version).strip() + "\n")

def _parent_of(version: str, lineage: Dict[str, Any]) -> Optional[str]:
    vs = _lineage_versions(lineage)
    try:
        idx = vs.index(str(version))
        if idx > 0:
            return vs[idx-1]
    except ValueError:
        pass
    return vs[0] if vs else None

# --- decision & application ---

def _passes_guardrails(metrics: Dict[str, Any], g: Guardrails) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ok = True

    labels = int(metrics.get("labels", 0) or 0)
    precision = float(metrics.get("precision", 0.0) or 0.0)
    ece = float(metrics.get("ece", 1.0) or 1.0)

    if labels < g.min_labels:
        ok = False; reasons.append("labels_lt_min")
    else:
        reasons.append("labels_ok")

    if precision < g.min_precision:
        ok = False; reasons.append("precision_lt_min")
    else:
        reasons.append("precision_ok")

    if ece > g.max_ece:
        ok = False; reasons.append("ece_gt_max")
    else:
        reasons.append("ece_ok")

    return ok, reasons

def _apply_action(models_dir: Path, action: str, version: str, lineage: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    if action == "promote":
        _write_current_model(models_dir, version)
        return True, version
    elif action == "rollback":
        parent = _parent_of(version, lineage)
        if parent:
            _write_current_model(models_dir, parent)
            return True, parent
        return False, None
    return False, None

def _demo_seed(applied: List[Dict[str, Any]], skipped: List[Dict[str, Any]], have_any: bool) -> None:
    if not have_any:
        applied.append({
            "version": "v0.7.7", "action":"promote",
            "reason":["precision_improving","ece_ok"], "confidence": 0.91
        })
    if not skipped:
        skipped.append({
            "version": "v0.7.5", "action":"rollback",
            "reason":["cooldown"], "confidence": 0.82
        })

def _make_reversal_plan() -> Dict[str, Any]:
    return {
        "window_hours": 12,
        "trigger": "precision_regression|-0.02|any"
    }

# --- plotting (minimal, resilient) ---

def _plot_timeline(arts: Path, applied: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
        _ensure_dir(arts)
        xs: List[float] = []
        ys: List[float] = []
        labels: List[str] = []

        i = 0
        for a in applied:
            xs.append(i); ys.append(1.0); labels.append(f"{a['action']}:{a['version']}")
            i += 1
        for s in skipped:
            xs.append(i); ys.append(0.0); labels.append(f"{s['action']}:{s['version']}")
            i += 1

        plt.figure()
        plt.scatter(xs, ys)
        for x, y, lab in zip(xs, ys, labels):
            plt.text(x, y + 0.03, lab, fontsize=8, rotation=30)
        plt.yticks([0.0, 1.0], ["skipped", "applied"])
        plt.title("Governance Apply Timeline")
        plt.xlabel("events")
        plt.tight_layout()
        out = arts / "governance_apply_timeline.png"
        plt.savefig(out)
        plt.close()
    except Exception:
        out = arts / "governance_apply_timeline.png"
        _ensure_dir(arts)
        try:
            out.write_bytes(b"\x89PNG\r\n\x1a\n")
        except Exception:
            pass

# --- Public API ---

def append(md: List[str], ctx: Any) -> None:
    """
    Evaluate governance actions under guardrails & cooldown; optionally apply.
    Emits: models/governance_apply_result.json, logs/governance_apply.jsonl (append),
           artifacts/governance_apply_timeline.png, and a CI markdown block line.
    """
    models = Path(getattr(ctx, "models_dir", "models") or "models")
    arts = Path(getattr(ctx, "artifacts_dir", "artifacts") or "artifacts")
    logs = Path(getattr(ctx, "logs_dir", "logs") or "logs")
    _ensure_dir(models); _ensure_dir(arts); _ensure_dir(logs)

    plan = _read_json(models / "model_governance_actions.json") or {}
    lineage = _read_json(models / "model_lineage.json") or {}
    trend = _read_json(models / "model_performance_trend.json") or {}

    actions = plan.get("actions", []) if isinstance(plan.get("actions"), list) else []

    mode = (_env("MW_GOV_APPLY_MODE", "dryrun") or "dryrun").strip().lower()
    if mode not in ("dryrun","apply"):
        mode = "dryrun"
    demo = _env_bool("DEMO_MODE", False) or _env_bool("MW_DEMO", False)

    guards = _guardrails_from_env()

    ledger_path = logs / "governance_apply.jsonl"
    ledger_rows = _load_jsonl_latest(ledger_path)

    applied: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    current_before = _current_model(models_dir=models)

    for a in actions:
        ver = str(a.get("version") or "")
        act = str(a.get("action") or "")
        conf = float(a.get("confidence", 0.0) or 0.0)

        hours_since = _last_action_hours(ledger_rows, ver)
        if hours_since is not None and hours_since < guards.cooldown_h:
            skipped.append({"version": ver, "action": act, "reason": ["cooldown"], "confidence": conf})
            continue

        metrics = _metrics_for_version(trend, ver)
        ok, reasons = _passes_guardrails(metrics, guards)
        if not ok:
            skipped.append({"version": ver, "action": act, "reason": reasons, "confidence": conf})
            continue

        if act == "observe":
            applied.append({"version": ver, "action": act, "reason": reasons, "confidence": conf})
            continue

        if mode == "apply":
            did, newv = _apply_action(models, act, ver, lineage)
            if did:
                applied.append({"version": ver, "action": act, "reason": reasons, "confidence": conf})
                _append_jsonl(ledger_path, {
                    "ts": _now_ts(),
                    "at": _iso(),
                    "mode": mode,
                    "decision": "applied",
                    "action": act,
                    "version": ver,
                    "new_current": newv,
                    "reversal_window_h": 12.0,
                    "confidence": conf,
                    "reasons": reasons,
                })
            else:
                skipped.append({"version": ver, "action": act, "reason": reasons + (["no_parent"] if act=="rollback" else []), "confidence": conf})
        else:
            applied.append({"version": ver, "action": act, "reason": reasons, "confidence": conf})

    if demo:
        _demo_seed(applied, skipped, have_any=bool(applied))

    reversal = _make_reversal_plan()

    result = {
        "generated_at": _iso(),
        "mode": mode,
        "applied": applied,
        "skipped": skipped,
        "reversal_plan": reversal,
        "current_model_before": current_before,
        "current_model_after": _current_model(models),
    }
    (models / "governance_apply_result.json").write_text(json.dumps(result, indent=2))

    _plot_timeline(arts, applied, skipped)

    def _fmt_applied() -> str:
        if not applied:
            return "none"
        return ", ".join(f"{x['version']} {x['action']} ({x.get('confidence',0.0):.2f})" for x in applied)

    def _fmt_skipped() -> str:
        if not skipped:
            return "none"
        return ", ".join(f"{x['version']} {x['action']} [{','.join(x.get('reason',[]))}]" for x in skipped)

    # IMPORTANT: no leading newline so tests can match startswith("🧭 Governance Apply")
    md.append("🧭 Governance Apply (72 h, mode: %s)" % mode)
    md.append(f"applied: {_fmt_applied()} | skipped: {_fmt_skipped()}")
    md.append(f"reversal plan: monitor {reversal['window_hours']} h for precision regression ≥ 0.02")
