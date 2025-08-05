# tests/test_adaptive_threshold.py

import pytest
from src.adjustment_trigger_router import get_adaptive_threshold

@pytest.mark.parametrize("weight,expected", [
    (1.25, 0.4),   # high-trust reviewers now unsuppress at 0.4
    (1.1,  0.7),   # mid-tier reviewers use default 0.7
    (0.85, 0.8),   # low-trust boundary uses 0.8
    (0.5,  0.8),   # low-trust reviewers need stronger evidence at 0.8
])
def test_get_adaptive_threshold(weight, expected):
    """
    Verify that adaptive thresholds match the tiered logic:
      - weight >= 1.25 → threshold of 0.4
      - 0.85 < weight < 1.25 → threshold of 0.7
      - weight <= 0.85 → threshold of 0.8
    """
    assert get_adaptive_threshold(weight) == pytest.approx(expected)