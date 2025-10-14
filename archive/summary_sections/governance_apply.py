"""
Summary section forwarder for Governance Apply.
Keeps the registry decoupled from the governance module location.
"""

from typing import List
from .common import SummaryContext

def append(md: List[str], ctx: SummaryContext) -> None:
    # Import here to avoid import-time side-effects or circulars
    from scripts.governance.governance_apply import append as _append
    _append(md, ctx)
