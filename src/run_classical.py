"""Fit SARIMAX and Prophet for every city and score them against persistence.
Run from the repo root:
    python -m src.run_classical
SARIMAX settings are chosen on validation only; Prophet settings are fixed in advance.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.baselines import persistence
from src.classical import prophet_city_forecasts, sarimax_city_forecasts
from src.evaluation import evaluate

TABLES_DIR = config.ROOT / "reports" / "tables"
PRED_COLS = {
    "persistence": "pred_persistence",
    "sarimax_fourier": "pred_sarimax",
    "prophet_lag1": "pred_prophet_lag1",
}


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROCESSED / "aqi_clean.csv", parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def attach(df: pd.DataFrame, city: str, preds: pd.Series) -> pd.DataFrame:
    """Place a city's forecasts (indexed by date) into the full frame."""
    m = df["City"] == city
    df.loc[m, preds.name] = df.loc[m, "Date"].map(preds).to_numpy()
    return df


def plot_forecasts(df: pd.DataFrame, city: str = "Delhi", split: str = "val", days: int = 90) -> None:
    g = df[(df["City"] == city) & (df["split"] == split)].head(days)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(g["Date"], g[config.TARGET], color="black", label="actual")
    ax.plot(g["Date"], g["pred_persistence"], alpha=0.7, label="persistence")
    ax.plot(g["Date"], g["pred_sarimax"], alpha=0.9, label="SARIMAX (Fourier)")
    ax.plot(g["Date"], g["pred_prophet_lag1"], alpha=0.9, label="Prophet (lag-1)")
    ax.set_ylabel("AQI")
    ax.set_title(f"{city}: next-day forecasts, first {days} days of {split}")
    ax.legend()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"classical_{city.lower()}_{split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_clean()
    df["pred_persistence"] = persistence(df)
    df["pred_sarimax"] = float("nan")
    df["pred_prophet_lag1"] = float("nan")

    selections = []
    for city in config.CITIES:
        print(f"Fitting {city} ...")
        pred, table = sarimax_city_forecasts(df, city)
        df = attach(df, city, pred)
        selections.append(table)
        df = attach(df, city, prophet_city_forecasts(df, city))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    sel = pd.concat(selections, ignore_index=True)
    sel.to_csv(TABLES_DIR / "sarimax_selection.csv", index=False)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_PROCESSED / "preds_classical.csv", index=False)

    results = pd.concat([evaluate(df, PRED_COLS, s) for s in ("val", "test")], ignore_index=True)
    results.to_csv(TABLES_DIR / "classical_results.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== SARIMAX settings chosen on validation ===")
    print(sel[sel["chosen"]][["City", "order", "K", "val_MAE"]].round(1).to_string(index=False))
    for split in ("val", "test"):
        tab = results[results["split"] == split].pivot(index="City", columns="model", values="MAE")[list(PRED_COLS)]
        print(f"\n=== MAE by city ({split}) ===")
        print(tab.round(1).to_string())
        print(f"--- improvement over persistence, % (positive = better) ---")
        print((100 * (1 - tab.div(tab["persistence"], axis=0))).drop(columns="persistence").round(1).to_string())
    print("\n=== Pooled metrics ===")
    print(results[results["City"] == "ALL"].round(2).to_string(index=False))

    plot_forecasts(df, "Delhi", "val")
    print(f"\nTables saved to {TABLES_DIR}")


if __name__ == "__main__":
    main()