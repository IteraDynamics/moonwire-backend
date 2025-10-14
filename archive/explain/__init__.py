# scripts/explain/__init__.py

# Re-export the public explainer so callers can do:
#   from scripts.explain import explain_trigger
from .explain_trigger import (
    explain_trigger,
    explain_logistic,
    explain_tree,
    ExplainConfig,
)
__all__ = ["explain_trigger", "explain_logistic", "explain_tree", "ExplainConfig"]
