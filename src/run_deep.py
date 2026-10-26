"""Fit the LSTM and GRU models and score them against persistence. Run from the repo root:
    python -m src.run_deep
Configurations and epoch counts are chosen on validation only.
"""
import time

import pandas as pd

from src import config
from src.baselines import persistence
from src.deep_models import run_deep_models
from src.eda import core_pollutants, missing_pct
from src.evaluation import evaluate
from src.run_baselines import SEVERE_AQI
from src.split import get_split

TABLES_DIR = config.ROOT / "reports" / "tables"
PRED_COLS = {
    "persistence": "pred_persistence",
    "lstm": "pred_lstm",
    "gru": "pred_gru",
}


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROCESSED / "aqi_clean.csv", parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def main() -> None:
    df = load_clean()
    core = core_pollutants(missing_pct(get_split(df, "train")))      # train-only rule
    print("Core pollutants:", core)

    start = time.time()
    preds, table, chosen = run_deep_models(df, core)
    print(f"Trained in {time.time() - start:.0f}s")
    out = pd.concat([df, preds], axis=1)
    out["pred_persistence"] = persistence(out)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "deep_selection.csv", index=False)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.DATA_PROCESSED / "preds_deep.csv", index=False)

    results = pd.concat([evaluate(out, PRED_COLS, s) for s in ("val", "test")], ignore_index=True)
    results.to_csv(TABLES_DIR / "deep_results.csv", index=False)
    severe = pd.concat([evaluate(out, PRED_COLS, s, min_true_aqi=SEVERE_AQI) for s in ("val", "test")],
                       ignore_index=True)
    severe.to_csv(TABLES_DIR / "deep_results_severe.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Validation selection ===")
    print(table.round(3).to_string(index=False))
    print("\n=== Chosen on validation ===")
    for cell, c in chosen.items():
        print(f"{cell:5s} {c}")
    for split in ("val", "test"):
        tab = results[results["split"] == split].pivot(index="City", columns="model", values="MAE")[list(PRED_COLS)]
        print(f"\n=== MAE by city ({split}) ===")
        print(tab.round(1).to_string())
        print("--- improvement over persistence, % (positive = better) ---")
        print((100 * (1 - tab.div(tab["persistence"], axis=0))).drop(columns="persistence").round(1).to_string())
    print("\n=== Pooled metrics ===")
    print(results[results["City"] == "ALL"].round(2).to_string(index=False))
    print(f"\n=== MAE on severe days (actual AQI >= {SEVERE_AQI}) ===")
    for split in ("val", "test"):
        sv = severe[severe["split"] == split]
        print(f"--- {split} ---")
        print(sv.pivot(index="City", columns="model", values="MAE")[list(PRED_COLS)].round(1).to_string())
    print(f"\nTables saved to {TABLES_DIR}")


if __name__ == "__main__":
    main()
