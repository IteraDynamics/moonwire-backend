# tests/test_threshold_backtest.py
from datetime import datetime, timedelta, timezone
from pathlib import Path
import importlib, json, os

from scripts.summary_sections.common import SummaryContext

def _iso(dt):
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def test_threshold_backtest_math_and_artifact(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_THR_BT_WINDOW_H", "72")
    monkeypatch.setenv("MW_THRESHOLD_JOIN_MIN", "5")
    monkeypatch.setenv("MW_THR_BT_MIN_LABELS", "3")

    # current thresholds
    (models / "per_origin_thresholds.json").write_text(json.dumps({"reddit": 0.50, "twitter": 0.50}), encoding="utf-8")

    # recommendations artifact (from v0.6.0)
    rec = {
        "objective": {"type":"precision_min_recall_max","min_precision":0.75},
        "guardrails": {"max_delta":0.10, "allow_large_jump": False, "min_labels": 10},
        "per_origin": [
            {"origin":"reddit","recommended":0.56,"status":"ok"},
            {"origin":"twitter","recommended":0.47,"status":"ok"},
        ],
        "demo": False,
    }
    (models / "threshold_recommendations.json").write_text(json.dumps(rec), encoding="utf-8")

    now = datetime.now(timezone.utc)

    # Seed triggers with scores near labels
    # reddit: 3 labels: T at 0.80, T at 0.70, F at 0.60
    trig = [
        {"timestamp": _iso(now - timedelta(minutes=50)), "origin": "reddit", "adjusted_score": 0.80},
        {"timestamp": _iso(now - timedelta(minutes=49)), "origin": "reddit", "adjusted_score": 0.70},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "reddit", "adjusted_score": 0.60},

        {"timestamp": _iso(now - timedelta(minutes=40)), "origin": "twitter", "adjusted_score": 0.72},
        {"timestamp": _iso(now - timedelta(minutes=39)), "origin": "twitter", "adjusted_score": 0.50},
        {"timestamp": _iso(now - timedelta(minutes=38)), "origin": "twitter", "adjusted_score": 0.30},
    ]
    (models / "trigger_history.jsonl").write_text("\n".join(json.dumps(r) for r in trig), encoding="utf-8")

    labs = [
        {"timestamp": _iso(now - timedelta(minutes=50)), "origin": "reddit", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=49)), "origin": "reddit", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=48)), "origin": "reddit", "label": False},

        {"timestamp": _iso(now - timedelta(minutes=40)), "origin": "twitter", "label": True},
        {"timestamp": _iso(now - timedelta(minutes=39)), "origin": "twitter", "label": False},
        {"timestamp": _iso(now - timedelta(minutes=38)), "origin": "twitter", "label": True},
    ]
    (models / "label_feedback.jsonl").write_text("\n".join(json.dumps(r) for r in labs), encoding="utf-8")

    from scripts.summary_sections import threshold_backtest as tbt
    importlib.reload(tbt)

    ctx = SummaryContext(
        logs_dir=tmp_path / "logs",
        models_dir=models,
        is_demo=False,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )
    md: list[str] = []
    tbt.append(md, ctx)

    out = "\n".join(md)
    assert "Threshold Backtest" in out
    assert "reddit" in out and "twitter" in out

    # Check artifact exists with expected schema
    data = json.loads((models / "threshold_backtest.json").read_text(encoding="utf-8"))
    assert isinstance(data.get("per_origin"), list)
    ro = {r["origin"]: r for r in data["per_origin"]}
    # sanity: labels counted
    assert ro["reddit"]["labels"] == 3
    assert ro["twitter"]["labels"] == 3
    # deltas present
    for k in ("precision","recall","f1","triggers"):
        assert k in ro["reddit"]["delta"]
        assert k in ro["twitter"]["delta"]


def test_threshold_backtest_demo_fallback(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_THR_BT_WINDOW_H", "72")
    monkeypatch.setenv("DEMO_MODE", "true")

    # No logs, force demo
    from scripts.summary_sections import threshold_backtest as tbt
    importlib.reload(tbt)

    ctx = SummaryContext(
        logs_dir=tmp_path / "logs",
        models_dir=models,
        is_demo=True,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )
    md: list[str] = []
    tbt.append(md, ctx)
    text = "\n".join(md)
    assert "(demo)" in text
    data = json.loads((models / "threshold_backtest.json").read_text(encoding="utf-8"))
    assert data.get("demo") is True
    assert len(data.get("per_origin") or []) >= 1
