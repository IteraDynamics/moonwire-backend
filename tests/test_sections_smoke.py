from scripts.summary_sections.common import SummaryContext
from pathlib import Path

def test_sections_compile():
    from scripts.summary_sections import header_overview, source_yield_plan
    ctx = SummaryContext(logs_dir=Path("logs"), models_dir=Path("models"), is_demo=True)
    md = []
    header_overview.append(md, ctx, reviewers=[], threshold=2.5, sig_id="demo", triggered_log=[])
    source_yield_plan.append(md, ctx)
    assert md, "sections should append some text"