"""Exploratory analysis on the TRAIN split only (validation and test are never read
for any decision). Run from the repo root:
    python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")  # save figures to disk, no window needed
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.split import get_split

TABLES_DIR = config.ROOT / "reports" / "tables"
MAX_MISSING_PCT = 25.0                 # a feature must be this complete in EVERY city
ACF_LAGS = [1, 2, 3, 7, 14, 30, 365]


def load_train() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROCESSED / "aqi_clean.csv", parse_dates=["Date"])
    train = get_split(df, "train")
    _check_daily(train)
    return train


def _check_daily(train: pd.DataFrame) -> None:
    """Lags below count rows, so every city must have one row per consecutive day."""
    for city, g in train.groupby("City"):
        if not (g["Date"].diff().dropna().dt.total_seconds() == 86400).all():
            raise ValueError(f"{city}: train dates are not consecutive days")


def missing_pct(train: pd.DataFrame) -> pd.DataFrame:
    """% of train rows that are NaN, per column (rows) and city (columns)."""
    cols = config.POLLUTANTS + [config.TARGET]
    return train.groupby("City")[cols].apply(lambda g: g.isna().mean() * 100).T.round(1)


def zero_pct(train: pd.DataFrame) -> pd.DataFrame:
    """% of train rows where a pollutant is exactly 0 (possible placeholder)."""
    return train.groupby("City")[config.POLLUTANTS].apply(lambda g: (g == 0).mean() * 100).T.round(1)


def core_pollutants(miss: pd.DataFrame, max_missing_pct: float = MAX_MISSING_PCT) -> list:
    """Pollutants whose train missingness is within the limit in every city."""
    p = miss.loc[config.POLLUTANTS]
    return p.index[p.max(axis=1) <= max_missing_pct].tolist()


def autocorr_table(train: pd.DataFrame, lags=ACF_LAGS) -> pd.DataFrame:
    """Autocorrelation of AQI at each lag (in days), per city."""
    out = {}
    for city, g in train.groupby("City"):
        s = g.set_index("Date")[config.TARGET]
        out[city] = {lag: s.autocorr(lag) for lag in lags}
    return pd.DataFrame(out).round(2)


def lagged_corr(train: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """Correlation of AQI today with each pollutant `lag` days EARLIER, which is the
    only pollutant information available when forecasting."""
    shifted = train.groupby("City")[config.POLLUTANTS].shift(lag)
    out = {}
    for city in train["City"].unique():
        m = train["City"] == city
        out[city] = shifted[m].corrwith(train.loc[m, config.TARGET])
    return pd.DataFrame(out).round(2)


def monthly_profile(train: pd.DataFrame) -> pd.DataFrame:
    return train.assign(month=train["Date"].dt.month).pivot_table(
        index="month", columns="City", values=config.TARGET).round(0)


def yearly_profile(train: pd.DataFrame) -> pd.DataFrame:
    return train.assign(year=train["Date"].dt.year).pivot_table(
        index="year", columns="City", values=config.TARGET).round(0)


def weekday_profile(train: pd.DataFrame) -> pd.DataFrame:
    return train.assign(weekday=train["Date"].dt.dayofweek).pivot_table(
        index="weekday", columns="City", values=config.TARGET).round(0)


def _save(fig, name: str) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aqi_distribution(train: pd.DataFrame) -> None:
    cities = sorted(train["City"].unique())
    data = [train.loc[train["City"] == c, config.TARGET].dropna() for c in cities]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.boxplot(data, tick_labels=cities)
    ax.set_ylabel("AQI")
    ax.set_title("AQI distribution by city (train, 2015-2018)")
    _save(fig, "eda_aqi_distribution.png")


def plot_monthly_profile(train: pd.DataFrame) -> None:
    prof = monthly_profile(train)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for city in prof.columns:
        ax.plot(prof.index, prof[city], marker="o", label=city)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean AQI")
    ax.set_title("Seasonality: mean AQI by month (train)")
    ax.legend()
    _save(fig, "eda_monthly_profile.png")


def plot_missing(miss: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    im = ax.imshow(miss.values, aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(miss.columns)), miss.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(miss.index)), miss.index)
    for i in range(miss.shape[0]):
        for j in range(miss.shape[1]):
            ax.text(j, i, f"{miss.iat[i, j]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title("Missing % in train (after causal fill)")
    fig.colorbar(im, ax=ax, label="% missing")
    _save(fig, "eda_missing.png")


def plot_autocorr(train: pd.DataFrame, max_lag: int = 60) -> None:
    acf = autocorr_table(train, lags=range(1, max_lag + 1))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for city in acf.columns:
        ax.plot(acf.index, acf[city], label=city)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation of AQI")
    ax.set_title("How long AQI 'remembers' (train)")
    ax.legend()
    _save(fig, "eda_autocorrelation.png")


def main() -> None:
    train = load_train()
    print(f"Train rows: {len(train)} ({train['Date'].min().date()} to {train['Date'].max().date()})")

    miss = missing_pct(train)
    tables = {
        "missing_pct": miss,
        "zero_pct": zero_pct(train),
        "autocorr": autocorr_table(train),
        "lagged_corr_lag1": lagged_corr(train, 1),
        "monthly_profile": monthly_profile(train),
        "yearly_profile": yearly_profile(train),
        "weekday_profile": weekday_profile(train),
    }
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, tab in tables.items():
        tab.to_csv(TABLES_DIR / f"{name}.csv")
        print(f"\n=== {name} ===")
        print(tab.to_string())

    print(f"\nCore pollutants (<= {MAX_MISSING_PCT:.0f}% missing in every city): {core_pollutants(miss)}")

    plot_aqi_distribution(train)
    plot_monthly_profile(train)
    plot_missing(miss)
    plot_autocorr(train)
    print(f"\nFigures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()