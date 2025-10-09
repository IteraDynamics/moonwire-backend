"""
Thin wrapper so governance can be referenced as a summary section module.
Keeps import boundaries clean: summary_sections → (delegates) → governance.
"""
from typing import List
from scripts.governance.model_governance_actions import append as _append_core

def append(md: List[str], ctx) -> None:
    _append_core(md, ctx)