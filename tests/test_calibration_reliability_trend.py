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
    models = tmp_path / "models"; models.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    arts = tmp_path / "artifacts"; arts.mkdir()
    monkeypatch.chdir(tmp_path)

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    trig, labs = [], []
    for h in range(3,0,-1):  # three buckets
        t = now - timedelta(hours=h)
        for i in range(10):
            tid = f"id{h}_{i}"
            trig.append({"id":tid,"origin":"reddit","score":0.8 if i>=5 else 0.2,
                         "timestamp":_iso(t)})
            labs.append({"id":tid,"label":i>=5,"timestamp":_iso(t)})

    _write_jsonl(logs/"trigger_history.jsonl", trig)
    _write_jsonl(logs/"label_feedback.jsonl", labs)

    from scripts.summary_sections import calibration_reliability_trend as crt
    importlib.reload(crt)

    ctx = SummaryContext(logs_dir=logs, models_dir=models, is_demo=False)
    md=[]
    crt.append(md,ctx)

    out="\n".join(md)
    assert "Calibration & Reliability Trend" in out
    assert "reddit" in out
    assert (models/"calibration_reliability_trend.json").exists()
    d=json.loads((models/"calibration_reliability_trend.json").read_text())
    assert "series" in d and d["series"]
    img1=arts/"calibration_trend_ece.png"
    img2=arts/"calibration_trend_brier.png"
    assert img1.exists() and img2.exists()


def test_demo_fallback(tmp_path, monkeypatch):
    models=tmp_path/"models"; models.mkdir()
    logs=tmp_path/"logs"; logs.mkdir()
    arts=tmp_path/"artifacts"; arts.mkdir()
    monkeypatch.setenv("DEMO_MODE","true")
    monkeypatch.chdir(tmp_path)

    from scripts.summary_sections import calibration_reliability_trend as crt
    importlib.reload(crt)
    ctx=SummaryContext(logs_dir=logs, models_dir=models, is_demo=True)
    md=[]
    crt.append(md,ctx)
    out="\n".join(md)
    assert "(demo)" in out
    d=json.loads((models/"calibration_reliability_trend.json").read_text())
    assert d.get("demo") is True
    assert d["series"]
    assert (arts/"calibration_trend_ece.png").exists()