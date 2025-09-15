# scripts/summary_sections/__init__.py
# marks "summary_sections" as a package
from .common import SummaryContext

# optional convenience re-exports (not required, but handy)
from . import score_distribution              # existing global dist section
from . import signal_quality_per_origin       # v0.5.6
from . import score_distribution_per_origin   # ⬅️ NEW v0.5.7

__all__ = [
    "SummaryContext",
    "score_distribution",
    "signal_quality_per_origin",
    "score_distribution_per_origin",  # ⬅️ NEW
]