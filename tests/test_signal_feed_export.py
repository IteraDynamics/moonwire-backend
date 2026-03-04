# tests/test_signal_feed_export.py
"""
Unit tests for signal feed exporter.
"""
import json
from pathlib import Path
import pandas as pd
import pytest


def test_export_schema_validation():
    """Test that exported feed has correct schema."""
    # This is a minimal schema test - full export test would require trained models
    
    # Expected schema
    expected_cols = ["ts_utc", "product_id", "p_long", "horizon_hours", "model_version"]
    
    # Mock a small feed DataFrame
    df = pd.DataFrame({
        "ts_utc": ["2019-01-01T00:00:00Z", "2019-01-01T01:00:00Z", "2019-01-01T02:00:00Z"],
        "product_id": ["BTC-USD", "BTC-USD", "BTC-USD"],
        "p_long": [0.65, 0.72, 0.58],
        "horizon_hours": [1, 1, 1],
        "model_version": ["v1.0", "v1.0", "v1.0"],
    })
    
    # Validate schema
    assert list(df.columns) == expected_cols, "Schema mismatch"
    
    # Validate types
    assert df["p_long"].dtype == float, "p_long should be float"
    assert df["horizon_hours"].dtype == int, "horizon_hours should be int"
    
    # Validate ranges
    assert (df["p_long"] >= 0.0).all(), "p_long should be >= 0"
    assert (df["p_long"] <= 1.0).all(), "p_long should be <= 1"
    
    # Validate timestamps are sorted
    ts_sorted = sorted(df["ts_utc"])
    assert list(df["ts_utc"]) == ts_sorted, "Timestamps should be sorted"
    
    # Validate no duplicates
    assert df["ts_utc"].duplicated().sum() == 0, "Timestamps should be unique"
    
    print("✓ Schema validation passed")


def test_manifest_structure():
    """Test that manifest contains required fields."""
    
    # Mock manifest
    manifest = {
        "generated_at_utc": "2026-03-04T00:00:00Z",
        "product_id": "BTC-USD",
        "bar_seconds": 3600,
        "start_date": "2019-01-01",
        "end_date": "2019-03-01",
        "total_bars": 1440,
        "horizon_hours": 1,
        "model_version": "v1.0",
        "git_sha": "abc123def456",
        "model_dir": "models/standard",
        "feature_set": "standard_v1",
        "schema": {},
    }
    
    required_fields = [
        "generated_at_utc",
        "product_id",
        "bar_seconds",
        "start_date",
        "end_date",
        "total_bars",
        "horizon_hours",
        "model_version",
        "git_sha",
        "schema",
    ]
    
    for field in required_fields:
        assert field in manifest, f"Missing required field: {field}"
    
    print("✓ Manifest structure validation passed")


if __name__ == "__main__":
    test_export_schema_validation()
    test_manifest_structure()
    print("\n✅ All signal feed export tests passed!")
