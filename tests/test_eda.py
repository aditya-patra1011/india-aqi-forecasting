import numpy as np
import pandas as pd
import pytest

from src import config
from src.eda import (_check_daily, autocorr_table, core_pollutants,
                     lagged_corr, missing_pct)


def _frame(city, n=200, **cols):
    df = pd.DataFrame({"City": city, "Date": pd.date_range("2018-01-01", periods=n)})
    for c in config.POLLUTANTS + [config.TARGET]:
        df[c] = cols.get(c, np.nan)
    return df


def test_lagged_corr_uses_yesterdays_pollutant_not_todays():
    x = np.random.default_rng(0).normal(size=200)
    aqi = np.r_[np.nan, x[:-1]]                      # AQI today == PM2.5 yesterday
    df = _frame("A", **{"PM2.5": x, "AQI": aqi})
    assert lagged_corr(df, 1).loc["PM2.5", "A"] == pytest.approx(1.0)
    assert abs(lagged_corr(df, 0).loc["PM2.5", "A"]) < 0.3


def test_lagged_corr_does_not_mix_cities():
    x = np.random.default_rng(1).normal(size=100)
    a = _frame("A", n=100, **{"PM2.5": x, "AQI": np.r_[np.nan, x[:-1]]})
    b = _frame("B", n=100, **{"PM2.5": x[::-1], "AQI": x})
    out = lagged_corr(pd.concat([a, b], ignore_index=True), 1)
    assert out.loc["PM2.5", "A"] == pytest.approx(1.0)
    assert out.loc["PM2.5", "B"] < 0.9


def test_autocorr_lag_indexing():
    s = np.tile([1.0, -1.0], 100)                    # alternating series
    out = autocorr_table(_frame("A", AQI=s), lags=[1, 2])
    assert out.loc[1, "A"] == pytest.approx(-1.0)
    assert out.loc[2, "A"] == pytest.approx(1.0)


def test_core_pollutants_drops_columns_missing_in_any_city():
    ok = np.arange(200.0)
    a = _frame("A", **{"PM2.5": ok, "PM10": ok})
    b = _frame("B", **{"PM2.5": ok})                  # PM10 entirely missing in city B
    miss = missing_pct(pd.concat([a, b], ignore_index=True))
    core = core_pollutants(miss)
    assert "PM2.5" in core
    assert "PM10" not in core


def test_check_daily_rejects_gaps():
    df = _frame("A", n=10)
    _check_daily(df)                                  # consecutive days: fine
    with pytest.raises(ValueError):
        _check_daily(df.drop(index=4))