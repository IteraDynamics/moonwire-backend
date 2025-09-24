# Make "scripts" a package.
# Avoid importing a non-existent "scripts.common".
# For convenience (and backward-compat), re-export SummaryContext from summary_sections.common if available.

try:
    from .summary_sections.common import SummaryContext  # noqa: F401
    __all__ = ["SummaryContext"]
except Exception:
    __all__ = []