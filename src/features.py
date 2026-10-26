"""Feature engineering for next-day AQI. The row for day d contains only information
from days before d (lags and trailing averages that stop at d - 1) plus calendar
features of day d itself, which are known in advance. Nothing from day d's own AQI
or pollutant readings is ever used."""
import numpy as np
import pandas as pd

from src import config
from src.baselines import check_continuous

AQI_LAGS = (1, 2, 3, 7)
ROLL_WINDOWS = (7, 30)
POLLUTANT_LAGS = (1,)

FEATURE_SETS = ("A_aqi_only", "B_core", "C_all")


def pollutants_for(feature_set: str, core) -> list:
    """Pollutants whose lagged values are included in each feature set."""
    sets = {
        "A_aqi_only": [],
        "B_core": list(core),                 # <= 25% missing in every city (train-only rule)
        "C_all": list(config.POLLUTANTS),
    }
    if feature_set not in sets:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return sets[feature_set]


def build_features(df: pd.DataFrame, pollutants) -> pd.DataFrame:
    """Feature matrix aligned with df's index. df must hold one row per consecutive
    day for each city, sorted by City and Date (as produced by src.make_dataset)."""
    check_continuous(df)
    g = df.groupby("City")
    cols = {}
    for lag in AQI_LAGS:
        cols[f"aqi_lag{lag}"] = g[config.TARGET].shift(lag)
    for w in ROLL_WINDOWS:
        cols[f"aqi_mean{w}"] = g[config.TARGET].transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=max(1, w // 2)).mean())
    for p in pollutants:
        for lag in POLLUTANT_LAGS:
            cols[f"{p}_lag{lag}"] = g[p].shift(lag)

    doy = df["Date"].dt.dayofyear
    cols["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    cols["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    cols["month"] = df["Date"].dt.month
    for city in sorted(df["City"].unique()):
        cols[f"city_{city}"] = (df["City"] == city).astype(int)
    return pd.DataFrame(cols, index=df.index)