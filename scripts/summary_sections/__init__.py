# marks "summary_sections" as a package
from .common import SummaryContext

# optional convenience re-exports
from . import score_distribution
from . import signal_quality_per_origin
from . import score_distribution_per_origin
from . import calibration_reliability_trend
from . import market_context  # ⬅️ NEW v0.6.6

__all__ = [
    "SummaryContext",
    "score_distribution",
    "signal_quality_per_origin",
    "score_distribution_per_origin",
    "calibration_reliability_trend",
    "market_context",  # ⬅️ NEW
]