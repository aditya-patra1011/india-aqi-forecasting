import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import FEATURE_SETS, build_features, pollutants_for


def _frame(n=60, cities=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for c in cities:
        d = pd.DataFrame({"City": c, "Date": pd.date_range("2019-01-01", periods=n)})
        d["AQI"] = rng.uniform(50, 300, n)
        for p in config.POLLUTANTS:
            d[p] = rng.uniform(1, 100, n)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def test_features_for_a_day_never_use_that_days_own_values():
    df = _frame()
    base = build_features(df, ["PM2.5"])
    k = 25                                                   # row k belongs to city A
    pert_df = df.copy()
    pert_df.loc[k, ["AQI", "PM2.5"]] *= 10
    pert = build_features(pert_df, ["PM2.5"])
    assert base.iloc[: k + 1].equals(pert.iloc[: k + 1])    # rows up to and including day k: unchanged
    assert not base.iloc[k + 1].equals(pert.iloc[k + 1])    # day k+1 sees yesterday's change


def test_lags_do_not_cross_cities():
    df = _frame()
    X = build_features(df, ["PM2.5"])
    first_b = df.index[df["City"] == "B"][0]
    assert np.isnan(X.loc[first_b, "aqi_lag1"])
    assert np.isnan(X.loc[first_b, "PM2.5_lag1"])
    assert np.isnan(X.loc[first_b, "aqi_mean7"])            # no history yet, not city A's data


def test_lag_and_trailing_mean_values_are_correct():
    df = _frame(cities=("A",))
    X = build_features(df, ["PM2.5"])
    t = 20
    assert X.loc[t, "aqi_lag1"] == df.loc[t - 1, "AQI"]
    assert X.loc[t, "aqi_lag7"] == df.loc[t - 7, "AQI"]
    assert X.loc[t, "PM2.5_lag1"] == df.loc[t - 1, "PM2.5"]
    assert X.loc[t, "aqi_mean7"] == pytest.approx(df.loc[t - 7: t - 1, "AQI"].mean())   # excludes day t


def test_calendar_and_city_columns_exist():
    X = build_features(_frame(), [])
    assert {"doy_sin", "doy_cos", "month", "city_A", "city_B"} <= set(X.columns)
    assert not any(c.endswith("_lag1") and c != "aqi_lag1" for c in X.columns)   # no pollutant columns


def test_build_features_rejects_gaps_and_unknown_feature_set():
    df = _frame()
    with pytest.raises(ValueError):
        build_features(df.drop(index=10), [])
    with pytest.raises(ValueError):
        pollutants_for("nonsense", [])
    assert pollutants_for("C_all", []) == config.POLLUTANTS
    assert set(FEATURE_SETS) == {"A_aqi_only", "B_core", "C_all"}