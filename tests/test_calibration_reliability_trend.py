import json, importlib, os
from pathlib import Path
from datetime import datetime, timezone, timedelta

from scripts.summary_sections.common import SummaryContext, _iso


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_trend_from_logs(tmp_path, monkeypatch):
    models = tmp_path / "models"; models.mkdir(parents=True, exist_ok=True)
    logs = tmp_path / "logs"; logs.mkdir(parents=True, exist_ok=True)
    arts = tmp_path / "artifacts"; arts.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    # tighten knobs so we’re sure to get 3 buckets
    monkeypatch.setenv("MW_CAL_TREND_WINDOW_H", "6")
    monkeypatch.setenv("MW_CAL_TREND_BUCKET_MIN", "120")  # 2h -> 3 buckets in 6h
    monkeypatch.setenv("MW_CAL_TREND_DIM", "origin")

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    trig, labs = [], []
    # three buckets: now-6h, now-4h, now-2h (using bucket floors)
    for h in (6, 4, 2):
        t = now - timedelta(hours=h)
        for i in range(10):
            tid = f"id{h}_{i}"
            trig.append({"id": tid, "origin": "reddit", "score": 0.8 if i >= 5 else 0.2,
                         "timestamp": _iso(t)})
            labs.append({"id": tid, "label": i >= 5, "timestamp": _iso(t)})

    _write_jsonl(logs / "trigger_history.jsonl", trig)
    _write_jsonl(logs / "label_feedback.jsonl", labs)

    from scripts.summary_sections import calibration_reliability_trend as crt
    importlib.reload(crt)

    ctx = SummaryContext(logs_dir=logs, models_dir=models, is_demo=False)
    md = []
    crt.append(md, ctx)

    out = "\n".join(md)
    assert "Calibration & Reliability Trend" in out
    assert "reddit" in out

    jpath = models / "calibration_reliability_trend.json"
    assert jpath.exists()
    data = json.loads(jpath.read_text())
    assert data.get("series")
    # ensure at least 2 points for reddit
    series = {s["key"]: s for s in data["series"]}
    assert "reddit" in series and len(series["reddit"]["points"]) >= 2

    img1 = arts / "calibration_trend_ece.png"
    img2 = arts / "calibration_trend_brier.png"
    assert img1.exists() and img1.stat().st_size > 0
    assert img2.exists() and img2.stat().st_size > 0


def test_demo_fallback(tmp_path, monkeypatch):
    models = tmp_path / "models"; models.mkdir(parents=True, exist_ok=True)
    logs = tmp_path / "logs"; logs.mkdir(parents=True, exist_ok=True)
    arts = tmp_path / "artifacts"; arts.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.chdir(tmp_path)

    # keep defaults; no logs -> demo synth kicks in
    from scripts.summary_sections import calibration_reliability_trend as crt
    importlib.reload(crt)

    ctx = SummaryContext(logs_dir=logs, models_dir=models, is_demo=True)
    md = []
    crt.append(md, ctx)

    out = "\n".join(md)
    assert "(demo)" in out

    d = json.loads((models / "calibration_reliability_trend.json").read_text())
    assert d.get("demo") is True and d.get("series")

    assert (arts / "calibration_trend_ece.png").exists()
    assert (arts / "calibration_trend_brier.png").exists()