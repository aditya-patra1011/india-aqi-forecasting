"""Build data/processed/aqi_clean.csv. Run from the repo root:
    python -m src.make_dataset
"""
from src import config
from src.cleaning import (aqi_to_bucket, causal_ffill, clip_aqi,
                          mask_invalid_pollutants, reindex_daily)
from src.data_loader import load_raw, select_cities
from src.split import assign_split


def build():
    df = select_cities(load_raw())
    df = reindex_daily(df)
    df = mask_invalid_pollutants(df)
    df = clip_aqi(df)
    df = causal_ffill(df, config.POLLUTANTS)
    df[config.BUCKET_COL] = aqi_to_bucket(df[config.TARGET])
    return assign_split(df)


def main():
    df = build()
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_PROCESSED / "aqi_clean.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")
    print("\nRows per city and split:")
    print(df.groupby(["City", "split"]).size().unstack())
    print("\nAQI missing (%) per city and split:")
    print(df.assign(m=df[config.TARGET].isna()).groupby(["City", "split"])["m"].mean().mul(100).round(1).unstack())
    print(f"\nAQI rows capped at 500: {int(df['AQI_capped'].sum())}")


if __name__ == "__main__":
    main()