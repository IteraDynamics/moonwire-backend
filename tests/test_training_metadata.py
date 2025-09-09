import json
from pathlib import Path
from src.ml import training_metadata


def test_save_and_load_training_metadata(tmp_path: Path):
    # redirect logs to a temp file
    log_file = tmp_path / "training_runs.jsonl"

    # dummy input data
    version = "v0.test"
    rows = 10
    origin_counts = {"reddit": 4, "twitter": 6}
    label_counts = {"true": 7, "false": 3}
    metrics = {
        "logistic": {"roc_auc": 0.91, "pr_auc": 0.88, "logloss": 0.42},
        "rf": {"roc_auc": 0.85, "pr_auc": 0.75, "logloss": 0.55},
        "gb": {"roc_auc": 0.88, "pr_auc": 0.78, "logloss": 0.52},
    }
    top_features = ["burst_z", "count_24h"]

    # write one metadata entry
    training_metadata.save_training_metadata(
        version=version,
        rows=rows,
        origin_counts=origin_counts,
        label_counts=label_counts,
        metrics=metrics,
        top_features=top_features,
        path=log_file,  # ⬅️ temp file, not the real one
    )

    # file should exist with exactly 1 line
    assert log_file.exists()
    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 1

    # load it back
    latest = training_metadata.load_latest_training_metadata(path=log_file)
    assert latest is not None
    assert latest["version"] == version
    assert latest["rows"] == rows
    assert latest["origin_counts"]["reddit"] == 4
    assert latest["label_counts"]["true"] == 7
    assert "logistic" in latest["metrics"]
    assert "burst_z" in latest["top_features"]
