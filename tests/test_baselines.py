import numpy as np
import pandas as pd
import pytest

from src.baselines import (add_baselines, check_continuous, moving_average,
                           monthly_climatology, persistence, seasonal_naive)


def _frame(city, values, start="2019-12-25", split=None):
    n = len(values)
    df = pd.DataFrame({"City": city, "Date": pd.date_range(start, periods=n), "AQI": values})
    df["split"] = split if split is not None else "train"
    return df


def test_persistence_is_yesterday_and_never_crosses_cities():
    a = _frame("A", [10.0, 20.0, 30.0])
    b = _frame("B", [100.0, 200.0, 300.0])
    df = pd.concat([a, b], ignore_index=True)
    out = persistence(df, 1).tolist()
    assert np.isnan(out[0]) and out[1:3] == [10.0, 20.0]       # city A
    assert np.isnan(out[3]) and out[4:6] == [100.0, 200.0]     # city B starts with NaN, not A's last value


def test_seasonal_naive_uses_value_one_period_earlier():
    df = _frame("A", list(range(1, 11)))                        # AQI = 1..10
    out = seasonal_naive(df, period=7, horizon=1).tolist()
    assert all(np.isnan(out[:7]))
    assert out[7:] == [1, 2, 3]


def test_seasonal_naive_rejects_period_shorter_than_horizon():
    with pytest.raises(ValueError):
        seasonal_naive(_frame("A", [1.0, 2.0, 3.0]), period=1, horizon=2)


def test_moving_average_excludes_the_day_being_predicted():
    values = [10.0] * 8 + [1000.0]                              # spike on the last day
    df = _frame("A", values)
    out = moving_average(df, window=7, horizon=1)
    assert out.iloc[8] == pytest.approx(10.0)                   # spike must not leak into its own prediction
    assert out.iloc[7] == pytest.approx(10.0)


def test_climatology_is_fitted_on_train_only():
    train = _frame("A", [100.0] * 31, start="2018-01-01", split="train")          # January, all 100
    val_a = _frame("A", [100.0] * 31, start="2019-01-01", split="val")
    val_b = _frame("A", [9999.0] * 31, start="2019-01-01", split="val")           # wildly different val values
    pred_a = monthly_climatology(pd.concat([train, val_a], ignore_index=True))
    pred_b = monthly_climatology(pd.concat([train, val_b], ignore_index=True))
    assert pred_a.tolist() == pred_b.tolist()                    # validation values cannot change the fit
    assert pred_a.iloc[-1] == pytest.approx(100.0)


def test_check_continuous_rejects_gaps_and_add_baselines_runs():
    df = _frame("A", [1.0, 2.0, 3.0, 4.0])
    check_continuous(df)
    with pytest.raises(ValueError):
        check_continuous(df.drop(index=2))
    out = add_baselines(df)
    assert {"pred_persistence", "pred_mean_7d", "pred_seasonal_7d",
            "pred_seasonal_365d", "pred_climatology"} <= set(out.columns)