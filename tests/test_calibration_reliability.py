# tests/test_calibration_reliability.py
from datetime import datetime, timezone, timedelta
from pathlib import Path
import importlib, json, os

from scripts.summary_sections.common import SummaryContext, _iso


def _row(ts, origin, version, score=None, label=None):
    d = {"timestamp": _iso(ts), "origin": origin}
    if version:
        d["model_version"] = version
    if score is not None:
        d["adjusted_score"] = score
    if label is not None:
        d["label"] = label
    return d


def test_calibration_reliability_chart_and_json(monkeypatch, tmp_path: Path):
    models = tmp_path / "models"
    logs = tmp_path / "logs"
    models.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("MODELS_DIR", str(models))
    monkeypatch.setenv("MW_CAL_WINDOW_H", "72")
    monkeypatch.setenv("MW_THRESHOLD_JOIN_MIN", "5")
    monkeypatch.setenv("MW_CAL_BINS", "5")
    monkeypatch.setenv("MW_CAL_MIN_LABELS", "5")
    monkeypatch.setenv("MW_CAL_MAX_ECE", "0.2")

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Seed two versions:
    # v_good: probs match labels fairly well
    # v_bad: probs ~0.9 but labels half positive -> high ECE
    trig_rows = []
    lab_rows = []

    # v_good (twitter)
    for i in range(8):
        t = now - timedelta(minutes=60 - i)
        p = 0.1 if i < 4 else 0.8
        trig_rows.append(_row(t, "twitter", "v_good", score=p))
        lab_rows.append(_row(t, "twitter", "v_good", label=(i >= 4)))

    # v_bad (reddit)
    for i in range(8):
        t = now - timedelta(minutes=120 - i)
        p = 0.9
        trig_rows.append(_row(t, "reddit", "v_bad", score=p))
        lab_rows.append(_row(t, "reddit", "v_bad", label=(i % 2 == 0)))  # ~50% positives

    (models / "trigger_history.jsonl").write_text("\n".join(json.dumps(r) for r in trig_rows), encoding="utf-8")
    (models / "label_feedback.jsonl").write_text("\n".join(json.dumps(r) for r in lab_rows), encoding="utf-8")

    (tmp_path / "artifacts").mkdir(exist_ok=True)  # where plots go

    from scripts.summary_sections import calibration_reliability as cr
    importlib.reload(cr)

    ctx = SummaryContext(
        logs_dir=logs,
        models_dir=models,
        is_demo=False,
        origins_rows=[],
        yield_data=None,
        candidates=[],
        caches={},
    )

    md: list[str] = []
    cr.append(md, ctx)

    txt = "\n".join(md)
    assert "Calibration & Reliability" in txt
    assert "v_good" in txt and "v_bad" in txt

    # JSON exists and has the two versions
    data = json.loads((models / "calibration_reliability.json").read_text(encoding="utf-8"))
    vers = {r["version"] for r in data.get("per_version", [])}
    assert {"v_good", "v_bad"} <= vers

    # Alerts: v_bad should be "high_ece" given max_ece=0.2; v_good should be ok/low numbers
    byv = {r["version"]: r for r in data["per_version"]}
    assert "high_ece" in byv["v_bad"]["alerts"]

    # Images created
    img1 = tmp_path / "artifacts" / "cal_reliability_v_good.png"
    img2 = tmp_path / "artifacts" / "cal_reliability_v_bad.png"
    assert img1.exists() and img1.stat().st_size > 0
    assert img2.exists() and img2.stat().st_size > 0