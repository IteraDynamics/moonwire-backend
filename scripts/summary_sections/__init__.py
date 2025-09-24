# scripts/summary_sections/__init__.py
"""
Summary sections registry.

Each section exposes:
    append(md: List[str], ctx: SummaryContext) -> None

This module imports sections so callers can reference them as:
    from scripts.summary_sections import market_context, calibration_reliability_trend, ...
"""

from .common import SummaryContext  # re-export

# Import sections here so they are discoverable to callers
from . import market_context
from . import calibration_reliability_trend