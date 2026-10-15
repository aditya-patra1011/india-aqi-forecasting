import numpy as np
import pandas as pd

from src import config
from src.cleaning import (aqi_to_bucket, causal_ffill, clip_aqi, 
                        mask_invalid_pollutants, reindex_daily)

def _frame(city, dates, **cols):
    df = pd.DataFrame({"City": city, "Date": pd.to_datetime(dates)})
    for c in config.POLLUTANTS + [config.TARGET]:
        df[c] = cols.get(c, np.nan)
    return df

def test_reindex_daily_adds_missing_days():
    df = _frame("A", ["2020-01-01", "2020-01-04"], AQI=[10.0, 20.0])
    out = reindex_daily(df)
    assert len(out) == 4
    assert out["City"].eq("A").all()
    assert out["AQI"].isna().sum() == 2

def test_mask_invalid_pollutants_sets_negative_to_nan():
    df = _frame("A", ["2020-01-01", "2020-01-02"], **{"PM2.5": [-5.0, 30.0]})
    out = mask_invalid_pollutants(df)
    assert np.isnan(out.loc[0, "PM2.5"])
    assert out.loc[1, "PM2.5"] == 30.0

def test_clip_aqi_caps_and_flags():
    df = _frame("A", ["2020-01-01", "2020-01-02", "2020-01-03"], AQI=[700.0, 500.0, np.nan])
    out = clip_aqi(df)
    assert out["AQI"].tolist()[:2] == [500.0, 500.0]
    assert out["AQI_capped"].tolist() == [True, False, False]
    assert np.isnan(out.loc[2, "AQI"])

def test_causal_ffill_respects_limit_and_never_ffils_backward():
    vals = [np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, 9.0]
    df = _frame("A", pd.date_range("2020-01-01", periods=7), **{"PM10": vals})
    out = causal_ffill(df, ["PM10"], limit=3)["PM10"].tolist()
    assert np.isnan(out[0])                 # Leading NaN stays NaN (no backward fill)
    assert out[1:5] == [1.0, 1.0, 1.0, 1.0] #Filled for 3 days only
    assert np.isnan(out[5])
    assert out[6] == 9.0

def test_causal_ffill_does_not_bleed_across_cities():
    a = _frame("A", ["2020-01-01"], **{"PM10": [50.0]})
    b = _frame("B", ["2020-01-01"], **{"PM10": [np.nan]})
    out = causal_ffill(pd.concat([a, b], ignore_index=True), ["PM10"])
    assert np.isnan(out.loc[out["City"] == "B", "PM10"].iloc[0])

def test_causal_ffill_leaves_target_untouched():
    df = _frame("A", pd.date_range("2020-01-01", periods=3), AQI=[100.0, np.nan, np.nan])
    out = causal_ffill(df, config.POLLUTANTS)
    assert out["AQI"].isna().sum() == 2

def test_aqi_to_bucket_boundaries():
    s = pd.Series([13, 50, 51, 100, 101, 200, 201, 300, 301, 400, 401, 500, np.nan])
    got = aqi_to_bucket(s).tolist()
    assert got[:12] == ["Good", "Good", "Satisfactory", "Satisfactory", "Moderate", "Moderate",
                        "Poor", "Poor", "Very Poor", "Very Poor", "Severe", "Severe"]
    assert pd.isna(got[12])