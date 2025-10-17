# scripts/ci_artifacts.py
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
from datetime import datetime, timezone

PNG_1x1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0bIDAT\x08\xd7c`\x00\x00"
    b"\x00\x02\x00\x01\x0e\xc2\x02\xbd\x00\x00\x00\x00IEND\xaeB`\x82"
)

def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_png(path: Path, label: str = "") -> None:
    if path.exists():
        return
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        ensure_dir(path.parent)
        fig = plt.figure(figsize=(3,2), dpi=100)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, label or "MoonWire", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(str(path))
        plt.close(fig)
    except Exception:
        ensure_dir(path.parent)
        path.write_bytes(PNG_1x1)

def write_json(path: Path, obj) -> None:
    if path.exists():
        return
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def seed_versioned_stub(models: Path, version: str = "v0.5.1") -> None:
    vdir = ensure_dir(models / version)
    # minimal presence
    write_json(vdir / "stub.meta.json", {
        "generated_at": _now(),
        "version": version,
        "kind": "demo_stub"
    })
    (vdir / "README.txt").write_text(
        f"MoonWire versioned stub for CI uploads.\nGenerated at {_now()}.\n"
    )

def seed_placeholders(root: Path) -> None:
    artifacts = ensure_dir(root / "artifacts")
    models    = ensure_dir(root / "models")
    logs      = ensure_dir(root / "logs")

    # Governance dashboard
    (artifacts / "governance_dashboard.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>MoonWire Governance Dashboard</title>"
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;padding:16px}</style>"
        "<h1>MoonWire Governance Dashboard</h1><p>Placeholder artifact (CI).</p>"
    ) if not (artifacts / "governance_dashboard.html").exists() else None
    write_png(artifacts / "governance_dashboard.png", "governance dashboard")
    write_json(models / "governance_dashboard_manifest.json", {
        "generated_at": _now(), "window_hours": 72, "demo": True
    })

    # Performance
    write_json(models / "performance_metrics.json", {
        "generated_at": _now(), "trades": 0, "sharpe": None, "sortino": None,
        "max_drawdown": None, "win_rate": None, "profit_factor": None, "by_symbol": {}
    })
    (artifacts / "perf_summary.txt").write_text(
        "No backtestable trades in demo window.\n"
    ) if not (artifacts / "perf_summary.txt").exists() else None
    write_png(artifacts / "perf_equity_curve.png", "equity curve (demo)")
    write_png(artifacts / "perf_heatmap.png", "perf heatmap (demo)")

    # Market
    write_json(models / "market_context.json", {
        "generated_at": _now(), "coins": [], "demo": True
    })
    write_png(artifacts / "market_trend_price_demo.png", "price trend (demo)")
    write_png(artifacts / "market_trend_returns_demo.png", "returns trend (demo)")

    # Calibration / reliability
    write_json(models / "calibration_reliability.json", {
        "generated_at": _now(), "bins": 10, "ece": None, "curves": [], "demo": True
    })
    write_json(models / "calibration_per_origin.json", {
        "generated_at": _now(), "origins": [], "demo": True
    })
    write_png(artifacts / "cal_reliability_demo.png", "calibration (demo)")

    # Correlation + lead-lag
    write_json(models / "cross_origin_correlation.json", {
        "generated_at": _now(), "pairs": [], "demo": True
    })
    write_png(artifacts / "corr_heatmap.png", "corr heatmap (demo)")
    write_png(artifacts / "corr_leadlag.png", "lead/lag (demo)")

    write_json(models / "leadlag_analysis.json", {
        "generated_at": _now(), "shifts": [], "demo": True
    })
    write_png(artifacts / "leadlag_heatmap.png", "leadlag heatmap (demo)")
    write_png(artifacts / "leadlag_ccf_demo.png", "ccf (demo)")

    # Lineage + registry
    write_json(models / "model_lineage.json", {
        "generated_at": _now(), "edges": [], "demo": True
    })
    write_png(artifacts / "model_lineage_graph.png", "lineage (demo)")
    write_json(models / "model_registry.json", {
        "generated_at": _now(), "models": [], "demo": True
    })

    # Performance trend
    write_json(models / "model_performance_trend.json", {
        "generated_at": _now(), "series": [], "demo": True
    })
    write_png(artifacts / "model_performance_trend_metrics.png", "perf trend (demo)")
    write_png(artifacts / "model_performance_trend_alerts.png", "alerts (demo)")

    # Drift / retrain
    write_json(models / "drift_response_plan.json", {
        "generated_at": _now(), "candidates": [], "demo": True
    })
    write_png(artifacts / "drift_response_timeline.png", "drift timeline (demo)")
    write_png(artifacts / "drift_response_backtest_demo.png", "drift backtest (demo)")

    write_json(models / "retrain_plan.json", {
        "generated_at": _now(), "candidates": [], "demo": True
    })
    write_png(artifacts / "retrain_eval_demo.png", "retrain eval (demo)")
    write_png(artifacts / "retrain_reliability_demo.png", "retrain reliability (demo)")
    write_png(artifacts / "retrain_confusion_demo.png", "retrain confusion (demo)")

    # Misc: CI summary should already exist; keep a placeholder if not.
    (artifacts / "demo_summary.md").write_text(
        "MoonWire CI Demo Summary\n\n*(placeholder)*\n"
    ) if not (artifacts / "demo_summary.md").exists() else None

    # Logs minimal
    (logs / "governance_actions.jsonl").write_text(
        json.dumps({"ts": _now(), "action": "demo_init"}) + "\n"
    ) if not (logs / "governance_actions.jsonl").exists() else None

    # Versioned stub for prior upload globs
    seed_versioned_stub(models, os.getenv("MODEL_VERSION", "v0.5.1"))

def write_manifest(root: Path) -> None:
    """
    Write artifacts/manifest.json with sizes + sha256 for key files.
    """
    artifacts = root / "artifacts"
    models    = root / "models"
    files: list[dict] = []

    def add(path: Path):
        if not path.exists() or not path.is_file():
            return
        rel = path.relative_to(root).as_posix()
        files.append({
            "path": rel,
            "size": path.stat().st_size,
            "sha256": sha256_of(path)
        })

    # Index artifacts folder and selected model/log files
    for p in artifacts.rglob("*"):
        if p.is_file():
            add(p)
    for p in [
        models / "governance_dashboard_manifest.json",
        models / "performance_metrics.json",
        models / "calibration_reliability.json",
        models / "calibration_per_origin.json",
        models / "cross_origin_correlation.json",
        models / "leadlag_analysis.json",
        models / "model_lineage.json",
        models / "model_performance_trend.json",
        models / "model_registry.json",
        models / "drift_response_plan.json",
        models / "retrain_plan.json",
    ]:
        add(p)

    manifest = {
        "generated_at": _now(),
        "git_sha": os.getenv("GITHUB_SHA"),
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "repository": os.getenv("GITHUB_REPOSITORY"),
        "count": len(files),
        "files": files,
    }
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    root = Path(".").resolve()
    seed_placeholders(root)
    write_manifest(root)
