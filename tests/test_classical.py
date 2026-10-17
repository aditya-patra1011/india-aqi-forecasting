import numpy as np
import pandas as pd
import pytest

from src.classical import (city_series, fit_sarimax, fourier_terms, one_step_forecasts,
                           prophet_city_forecasts, sarimax_city_forecasts, select_sarimax)

N_TRAIN, N_VAL, N_TEST = 800, 120, 60


def _city_frame(seed=0):
    """Synthetic city: yearly cycle plus AR(1) noise, always positive."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL + N_TEST
    t = np.arange(n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i - 1] + rng.normal(scale=0.15)
    aqi = np.exp(5.0 + 0.4 * np.sin(2 * np.pi * t / 365.25) + noise)
    split = ["train"] * N_TRAIN + ["val"] * N_VAL + ["test"] * N_TEST
    return pd.DataFrame({"City": "A", "Date": pd.date_range("2016-01-01", periods=n),
                         "AQI": aqi, "split": split})


def _perturbed(df, split, position, factor=3.0):
    out = df.copy()
    idx = out.index[out["split"] == split][position]
    out.loc[idx, "AQI"] *= factor
    return out


def test_fourier_terms_shape_and_range():
    f = fourier_terms(pd.date_range("2018-01-01", periods=10), k=3)
    assert f.shape == (10, 6)
    assert f.abs().to_numpy().max() <= 1.0
    assert fourier_terms(["2000-01-01"], 1).loc["2000-01-01", "sin1"] == pytest.approx(0.0)


def test_sarimax_one_step_forecasts_do_not_use_the_future():
    df = _city_frame()
    train, val = city_series(df, "A", ["train"]), city_series(df, "A", ["val"])
    res = fit_sarimax(train, (1, 0, 0), 1)
    base = one_step_forecasts(res, val, 1)
    k = 50
    val2 = val.copy()
    val2.iloc[k] *= 3
    pert = one_step_forecasts(res, val2, 1)
    assert (base.iloc[: k + 1] - pert.iloc[: k + 1]).abs().max() == 0.0   # up to day k: unchanged
    assert abs(base.iloc[k + 1] - pert.iloc[k + 1]) > 0                    # day k+1 sees the change


def test_sarimax_val_forecasts_ignore_test_data_and_test_forecasts_ignore_the_future():
    df = _city_frame()
    base, _ = sarimax_city_forecasts(df, "A", orders=[(1, 0, 0)], ks=[1])
    k = 30
    pert, _ = sarimax_city_forecasts(_perturbed(df, "test", k), "A", orders=[(1, 0, 0)], ks=[1])
    val_dates = df.loc[df["split"] == "val", "Date"]
    test_dates = df.loc[df["split"] == "test", "Date"].iloc[: k + 1]
    assert len(base) == N_VAL + N_TEST
    assert (base[val_dates] == pert[val_dates]).all()                      # val never sees test data
    assert (base[test_dates] == pert[test_dates]).all()                    # test forecasts up to day k unchanged


def test_prophet_val_forecasts_ignore_test_data_and_test_forecasts_ignore_the_future():
    df = _city_frame()
    base = prophet_city_forecasts(df, "A")
    k = 30
    pert = prophet_city_forecasts(_perturbed(df, "test", k), "A")
    val_dates = df.loc[df["split"] == "val", "Date"]
    test_dates = df.loc[df["split"] == "test", "Date"].iloc[: k + 1]
    assert len(base) == N_VAL + N_TEST
    assert np.allclose(base[val_dates], pert[val_dates])
    assert np.allclose(base[test_dates], pert[test_dates])


def test_select_sarimax_returns_lowest_val_mae_among_converged():
    df = _city_frame()
    train, val = city_series(df, "A", ["train"]), city_series(df, "A", ["val"])
    order, k, table = select_sarimax(train, val, orders=[(1, 0, 0), (2, 0, 0)], ks=[1, 2])
    ok = table[table["converged"]]
    best = ok.loc[ok["val_MAE"].idxmin()]
    assert str(order) == best["order"] and k == best["K"]
    assert len(table) == 4