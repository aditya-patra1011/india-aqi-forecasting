"""Fit the ML models and score them against persistence. Run from the repo root:
    python -m src.run_ml
Feature sets and hyperparameters are chosen on validation only.
"""
import pandas as pd

from src import config
from src.baselines import persistence
from src.eda import core_pollutants, missing_pct
from src.evaluation import evaluate
from src.ml_models import pooled_vs_per_city, run_models
from src.run_baselines import SEVERE_AQI
from src.split import get_split

TABLES_DIR = config.ROOT / "reports" / "tables"
PRED_COLS = {
    "persistence": "pred_persistence",
    "ridge": "pred_ridge",
    "random_forest": "pred_random_forest",
    "xgboost": "pred_xgboost",
}


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROCESSED / "aqi_clean.csv", parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def importance_table(final_models: dict, top: int = 10) -> pd.DataFrame:
    rows = []
    for kind in ("xgboost", "random_forest"):
        model, names = final_models[kind]
        imp = pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
        rows.append(imp.head(top).rename(kind).reset_index().rename(columns={"index": f"{kind}_feature"}))
    return pd.concat(rows, axis=1)


def main() -> None:
    df = load_clean()
    core = core_pollutants(missing_pct(get_split(df, "train")))      # train-only rule
    print("Core pollutants:", core)

    preds, table, chosen, final_models = run_models(df, core)
    out = pd.concat([df, preds], axis=1)
    out["pred_persistence"] = persistence(out)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLES_DIR / "ml_selection.csv", index=False)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.DATA_PROCESSED / "preds_ml.csv", index=False)

    results = pd.concat([evaluate(out, PRED_COLS, s) for s in ("val", "test")], ignore_index=True)
    results.to_csv(TABLES_DIR / "ml_results.csv", index=False)
    severe = pd.concat([evaluate(out, PRED_COLS, s, min_true_aqi=SEVERE_AQI) for s in ("val", "test")],
                       ignore_index=True)
    severe.to_csv(TABLES_DIR / "ml_results_severe.csv", index=False)
    pooling = pooled_vs_per_city(df, core, chosen)
    pooling.to_csv(TABLES_DIR / "ml_pooled_vs_per_city.csv", index=False)
    imp = importance_table(final_models)
    imp.to_csv(TABLES_DIR / "ml_feature_importance.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Chosen on validation ===")
    for kind, c in chosen.items():
        print(f"{kind:14s} {c['feature_set']:11s} {c['params']}")
    print("\n=== Validation pooled MAE by feature set (best params per cell) ===")
    print(table.pivot_table(index="model", columns="feature_set", values="val_MAE", aggfunc="min").round(1).to_string())
    for split in ("val", "test"):
        tab = results[results["split"] == split].pivot(index="City", columns="model", values="MAE")[list(PRED_COLS)]
        print(f"\n=== MAE by city ({split}) ===")
        print(tab.round(1).to_string())
        print("--- improvement over persistence, % (positive = better) ---")
        print((100 * (1 - tab.div(tab["persistence"], axis=0))).drop(columns="persistence").round(1).to_string())
    print("\n=== Pooled metrics ===")
    print(results[results["City"] == "ALL"].round(2).to_string(index=False))
    print(f"\n=== MAE on severe days (actual AQI >= {SEVERE_AQI}), with n ===")
    for split in ("val", "test"):
        sv = severe[severe["split"] == split]
        print(f"--- {split} ---")
        print(sv.pivot(index="City", columns="model", values="MAE")[list(PRED_COLS)].round(1).to_string())
        print("n days:", sv[sv["model"] == "persistence"].set_index("City")["n"].to_dict())
    print("\n=== Pooled vs per-city models (validation, chosen configs) ===")
    print(pooling.round(3).to_string(index=False))
    print("\n=== Top features (final models, fitted on train + validation) ===")
    print(imp.round(3).to_string(index=False))
    print(f"\nTables saved to {TABLES_DIR}")


if __name__ == "__main__":
    main()