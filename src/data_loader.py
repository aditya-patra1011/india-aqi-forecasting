"""
Loading and basic validation of the raw CPCB city-day file.
"""
import pandas as pd

from src import config

REQUIRED_COLUMNS = ["City", "Data", *config.POLLUTANTS, config.TARGET, config.BUCKET_COL]

def load_raw(path=config.DATA_RAW) -> pd.DataFrame:
    """Read the raw CSV and check that every expected column exists"""
    df = pd.read_csv(path, parse_dates=["Date"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in raw data: {missing}")
    return df

def select_cities(df: pd.DataFrame, cities=config.CITIES) -> pd.DataFrame:
    """Keep only the project cities: fail loudly on missing cities or duplicates."""
    out = df[df["City"].isin(cities)].copy()
    absent = sorted(set(cities) - set(out["City"].unique()))
    if absent:
        raise ValueError(f"Cities not found in data: {absent}")
    if out.duplicated(["City", "Date"]).any():
        raise ValueError("Duplicate (City, Date) rows found")
    return out.sort_values(["City", "Date"]).reset_index(drop=True)