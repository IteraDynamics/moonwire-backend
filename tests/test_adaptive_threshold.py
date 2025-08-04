import pytest
from src.adjustment_trigger_router import get_adaptive_threshold

@pytest.mark.parametrize("weight,expected", [
    (1.25, 0.6),   # high-trust
    (1.1,  0.7),   # mid-tier
    (0.85, 0.8),   # low-trust boundary
    (0.5,  0.8),   # low-trust
])
def test_get_adaptive_threshold(weight, expected):
    assert get_adaptive_threshold(weight) == pytest.approx(expected)
