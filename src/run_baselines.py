"""Score all baselines on validation and test. Run from the repo root:
    python -m src.run_baselines
The baselines have no tuned settings (7-day window and 365-day period are fixed in
advance), so reporting validation and test side by side involves no test-set selection.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.baselines import PRED_COLS, add_baselines
from src.evaluation import evaluate

TABLES_DIR = config.ROOT / "reports" / "tables"
SEVERE_AQI = 300  # CPCB "Very Poor" and above


def load_clean() -> pd.DataFrame:
    return pd.read_csv(config.DATA_PROCESSED / "aqi_clean.csv", parse_dates=["Date"])


def plot_mae(results: pd.DataFrame, split: str) -> None:
    tab = results[(results["split"] == split)].pivot(index="City", columns="model", values="MAE")
    tab = tab[list(PRED_COLS)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    tab.plot.bar(ax=ax)
    ax.set_ylabel("MAE (AQI points)")
    ax.set_title(f"Baseline MAE by city ({split} split)")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=0)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES_DIR / f"baseline_mae_{split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = add_baselines(load_clean())

    results = pd.concat([evaluate(df, PRED_COLS, s) for s in ("val", "test")], ignore_index=True)
    severe = pd.concat([evaluate(df, PRED_COLS, s, min_true_aqi=SEVERE_AQI) for s in ("val", "test")],
                       ignore_index=True)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLES_DIR / "baseline_results.csv", index=False)
    severe.to_csv(TABLES_DIR / "baseline_results_severe.csv", index=False)

    pd.set_option("display.width", 200)
    for split in ("val", "test"):
        print(f"\n=== MAE by city and baseline ({split}) ===")
        print(results[results["split"] == split].pivot(index="City", columns="model", values="MAE")
              [list(PRED_COLS)].round(1).to_string())
    print("\n=== Full metrics, ALL cities pooled ===")
    print(results[results["City"] == "ALL"].round(2).to_string(index=False))
    print(f"\n=== MAE on severe days (actual AQI >= {SEVERE_AQI}), with n ===")
    sv = severe[severe["split"] == "val"]
    print(sv.pivot(index="City", columns="model", values="MAE")[list(PRED_COLS)].round(1).to_string())
    print("n days:", sv[sv["model"] == "persistence"].set_index("City")["n"].to_dict())

    for split in ("val", "test"):
        plot_mae(results, split)
    print(f"\nTables saved to {TABLES_DIR}\nFigures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()