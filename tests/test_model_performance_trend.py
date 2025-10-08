from pathlib import Path
from scripts.summary_sections import model_performance_trend
from scripts.summary_sections.common import SummaryContext, ensure_dir, _read_json

def test_trend_demo(tmp_path:Path):
    mdir=tmp_path/"models"; adir=tmp_path/"artifacts"
    ensure_dir(mdir); ensure_dir(adir)
    ctx=SummaryContext(models_dir=mdir,artifacts_dir=adir)

    md=[]
    model_performance_trend.append(md,ctx)

    jpath=mdir/"model_performance_trend.json"
    assert jpath.exists()
    j=_read_json(jpath)
    assert isinstance(j.get("versions"),list) and len(j["versions"])>=3

    p1=adir/"model_performance_trend_metrics.png"
    p2=adir/"model_performance_trend_alerts.png"
    assert p1.exists() and p1.stat().st_size>1000
    assert p2.exists() and p2.stat().st_size>500

    block="\n".join(md)
    assert "Model Performance Trends" in block