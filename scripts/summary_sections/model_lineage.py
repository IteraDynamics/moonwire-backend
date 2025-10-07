# scripts/summary_sections/model_lineage.py
from __future__ import annotations

"""
Thin summary-section wrapper around governance.model_lineage.

Public API:
    append(md: List[str], ctx: SummaryContext) -> None
"""

from typing import List
from .common import SummaryContext
from scripts.governance import model_lineage as _lineage


def append(md: List[str], ctx: SummaryContext) -> None:
    """
    Delegate to governance.model_lineage.append to build JSON, PNG,
    and append the markdown block. Tolerant to missing inputs.
    """
    _lineage.append(md, ctx)
