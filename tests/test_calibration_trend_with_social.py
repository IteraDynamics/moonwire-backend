import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import importlib

from scripts.summary_sections import calibration_reliability_trend as crt

def _iso(dt):
    return dt.replace(microsecond=0).isoformat() + "Z"

def test_enrich_with_social_bursts(tmp_path):
    models = tmp_path / "models"; models.mkdir()
    arts = tmp_path / "arts"; arts.mkdir()

    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    buckets = [now - timedelta(hours=h) for h in (6, 4, 2)]

    trend = {
        "series": [
            {
                "key": "reddit",
                "points": [
                    {"bucket_start": _iso(buckets[0]), "ece": 0.05, "brier": 0.1, "alerts": ["high_ece"]},
                    {"bucket_start": _iso(buckets[1]), "ece": 0.06, "brier": 0.11, "alerts": []},
                    {"bucket_start": _iso(buckets[2]), "ece": 0.15, "brier": 0.18, "alerts": ["high_ece"]},
                ]
            }
        ]
    }
    (models / "calibration_reliability_trend.json").write_text(json.dumps(trend))

    bursts = {
        "bursts": [
            {"subreddit": "CryptoCurrency", "term": "ETF", "z": 2.5, "bucket_start": _iso(buckets[2])}
        ]
    }
    (models / "social_reddit_context.json").write_text(json.dumps(bursts))

    class Ctx:
        def __init__(self, models_dir, artifacts_dir):
            self.models_dir = models_dir
            self.artifacts_dir = artifacts_dir

    ctx = Ctx(models, arts)
    md = []
    importlib.reload(crt)
    crt.append(md, ctx)

    enriched = json.loads((models / "calibration_reliability_trend.json").read_text())
    pts = enriched["series"][0]["points"]
    assert pts[2]["social_bursts"]
    assert "social_burst_overlap" in pts[2]["alerts"]

    # check plots exist
    assert (arts / "calibration_trend_ece.png").exists()
    assert (arts / "calibration_trend_brier.png").exists()

    out = "\n".join(md)
    assert "Calibration & Reliability Trend vs Market + Social" in out
    assert "ETF" in out
