"""Cleaning steps. Every step is either a fixed rule or causal (uses only past
values), so nothing here is fitted on, or leaks information from, later dates."""
import pandas as pd

from src import config


def reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Give each city one row per calendar day (first to last date), so missing
    days become explicit NaN rows and later lag features count real days."""
    parts = []
    for city, g in df.groupby("City"):
        g = g.drop(columns="City").set_index("Date").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="D", name="Date")
        g = g.reindex(full)
        g.insert(0, "City", city)
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


AQI_MAX = 500  # the CPCB AQI scale tops out at 500


def mask_invalid_pollutants(df: pd.DataFrame) -> pd.DataFrame:
    """Negative concentrations are physically impossible: set them to NaN."""
    out = df.copy()
    cols = config.POLLUTANTS
    out[cols] = out[cols].mask(out[cols] < 0)
    return out


def clip_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Cap AQI at 500 and remember which rows were capped."""
    out = df.copy()
    out["AQI_capped"] = out[config.TARGET] > AQI_MAX
    out[config.TARGET] = out[config.TARGET].clip(upper=AQI_MAX)
    return out


FFILL_LIMIT_DAYS = 3  # fill pollutant gaps of at most 3 days, using the past only


def causal_ffill(df: pd.DataFrame, columns, limit: int = FFILL_LIMIT_DAYS) -> pd.DataFrame:
    """Forward-fill short gaps within each city. Forward-only, so a value is never
    filled from a later date. The target (AQI) is deliberately NOT filled."""
    out = df.sort_values(["City", "Date"]).copy()
    out[columns] = out.groupby("City")[columns].ffill(limit=limit)
    return out


BUCKET_BINS = [0, 50, 100, 200, 300, 400, 500]
BUCKET_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def aqi_to_bucket(aqi: pd.Series) -> pd.Series:
    """Map AQI (0-500) to the six CPCB categories; NaN stays NaN."""
    return pd.cut(aqi, bins=BUCKET_BINS, labels=BUCKET_LABELS, include_lowest=True).astype(object)