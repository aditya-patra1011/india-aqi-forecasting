import pandas as pd
import pytest
from src.split import assign_split, get_split

def _dates(*d):
    return pd.DataFrame({"Date": pd.to_datetime(list(d))})

def test_boundaries_are_assigned_correctly():
    df = _dates("2018-12-31", "2019-01-01", "2019-12-31", "2020-01-01")
    out = assign_split(df, "2018-12-31", "2019-12-31")
    assert out["split"].tolist() == ["train", "val", "val", "test"]

def test_splits_are_disjoint_ordered_and_completed():
    df = _dates(*pd.date_range("2015-01-01", "2020-07-01"))
    out = assign_split(df, "2018-12-31", "2019-12-31")
    tr, va, te = (get_split(out, n) for n in ("train", "val", "test"))
    assert len(tr) + len(va) +len(te) == len(df)
    assert tr["Date"].max() < va["Date"].min() <= va["Date"].max() < te["Date"].min()

def test_invalid_config_raises():
    with pytest.raises(ValueError):
        assign_split(_dates("2020-01-01"), "2019-12-31", "2018-12-31")
    with pytest.raises(ValueError):
        get_split(assign_split(_dates("2020-01-01")), "holdout")