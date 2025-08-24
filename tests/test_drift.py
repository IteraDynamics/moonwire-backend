from pathlib import Path
from src.ml.drift import compute_drift_from_stats

def test_drift_flags_large_shifts():
    # Training: mostly zeros
    train = {
        "count_6h": {"mean": 1.0, "nonzero_pct": 10.0, "min":0.0, "max": 3.0},
        "burst_z":  {"mean": 0.1, "nonzero_pct": 5.0,  "min":0.0, "max": 2.0},
    }
    # Recent: big means + nonzero jump
    recent = {
        "count_6h": {"mean": 6.0, "nonzero_pct": 90.0},
        "burst_z":  {"mean": 1.5, "nonzero_pct": 60.0},
    }
    ranked = compute_drift_from_stats(train, recent, ["count_6h","burst_z"], top=2)
    assert ranked and ranked[0]["feature"] in {"count_6h","burst_z"}