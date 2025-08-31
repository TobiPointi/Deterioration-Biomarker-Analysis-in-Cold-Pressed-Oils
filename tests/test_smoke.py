from biomarker_pipeline.scoring import minmax_01
import pandas as pd

def test_minmax_neutral_on_constant_series():
    s = pd.Series([1,1,1], index=list("abc"))
    out = minmax_01(s)
    assert (out == 0.5).all()
