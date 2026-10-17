"""Simple forecasting baselines for next-day AQI. Every model in later phases has
to beat these. Each baseline predicts the AQI of date d using only information
from dates up to d - HORIZON_DAYS, so none of them can see the day being predicted.

All functions expect the full cleaned frame (one row per city per consecutive day,
sorted by City and Date), as produced by src.make_dataset.
"""
import pandas as pd

from src import config

PRED_COLS = {
    "persistence": "pred_persistence",
    "mean_7d": "pred_mean_7d",
    "seasonal_naive_7d": "pred_seasonal_7d",
    "seasonal_naive_365d": "pred_seasonal_365d",
    "monthly_climatology": "pred_climatology",
}


def check_continuous(df: pd.DataFrame) -> None:
    """Shifts below count rows, so each city needs one row per consecutive day."""
    for city, g in df.groupby("City"):
        if not (g["Date"].diff().dropna().dt.total_seconds() == 86400).all():
            raise ValueError(f"{city}: dates are not consecutive days")


def persistence(df: pd.DataFrame, horizon: int = config.HORIZON_DAYS) -> pd.Series:
    """Prediction for day d = AQI on day d - horizon."""
    return df.groupby("City")[config.TARGET].shift(horizon)


def seasonal_naive(df: pd.DataFrame, period: int, horizon: int = config.HORIZON_DAYS) -> pd.Series:
    """Prediction for day d = AQI on day d - period (e.g. 7 = same weekday last week,
    365 = about the same date last year)."""
    if period < horizon:
        raise ValueError("period must be >= horizon, otherwise the baseline would peek ahead")
    return df.groupby("City")[config.TARGET].shift(period)


def moving_average(df: pd.DataFrame, window: int = 7, horizon: int = config.HORIZON_DAYS) -> pd.Series:
    """Prediction for day d = mean AQI of the `window` days ending at d - horizon."""
    return df.groupby("City")[config.TARGET].transform(
        lambda s: s.shift(horizon).rolling(window, min_periods=max(1, window // 2)).mean())


def monthly_climatology(df: pd.DataFrame) -> pd.Series:
    """Prediction for day d = average AQI of that calendar month in the TRAIN split,
    per city. This is the only fitted baseline, and it is fitted on train rows only."""
    train = df[df["split"] == "train"]
    clim = train.groupby(["City", train["Date"].dt.month])[config.TARGET].mean()
    idx = pd.MultiIndex.from_arrays([df["City"], df["Date"].dt.month])
    return pd.Series(clim.reindex(idx).to_numpy(), index=df.index)


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with one prediction column per baseline."""
    check_continuous(df)
    out = df.sort_values(["City", "Date"]).reset_index(drop=True)
    out[PRED_COLS["persistence"]] = persistence(out)
    out[PRED_COLS["mean_7d"]] = moving_average(out, 7)
    out[PRED_COLS["seasonal_naive_7d"]] = seasonal_naive(out, 7)
    out[PRED_COLS["seasonal_naive_365d"]] = seasonal_naive(out, 365)
    out[PRED_COLS["monthly_climatology"]] = monthly_climatology(out)
    return out