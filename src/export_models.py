"""Refit and save the models the app uses. Run from the repo root, after run_ml and
run_deep:
    python -m src.export_models
Both models are refitted on train + validation with the configuration chosen on
validation in phases 5 and 6, then checked against the test forecasts those phases
saved, so the app serves exactly the models that were evaluated.
"""
import ast

import numpy as np
import pandas as pd

from src import app_logic, config
from src.deep_models import fit_deep
from src.eda import core_pollutants, missing_pct
from src.features import build_features, pollutants_for
from src.ml_models import fit_predict
from src.split import get_split

TABLES_DIR = config.ROOT / "reports" / "tables"


def chosen_row(table: str, **match) -> pd.Series:
    t = pd.read_csv(TABLES_DIR / table)
    m = t["chosen"]
    for k, v in match.items():
        m &= t[k] == v
    return t[m].iloc[0]


def check_same(name: str, df: pd.DataFrame, pred: pd.Series, saved_file: str, col: str) -> None:
    saved = pd.read_csv(config.DATA_PROCESSED / saved_file, parse_dates=["Date"])
    merged = df[["City", "Date"]].assign(new=pred).merge(saved[["City", "Date", col]], on=["City", "Date"])
    test = merged[df["split"].to_numpy() == "test"]
    gap = float(np.nanmax(np.abs(test["new"] - test[col])))
    if not np.allclose(test["new"], test[col], atol=1e-3, equal_nan=True):
        raise RuntimeError(f"{name}: refitted test forecasts differ from {saved_file} (max gap {gap:.4f})")
    print(f"{name}: test forecasts match {saved_file} (max gap {gap:.2e})")


def main() -> None:
    df = app_logic.load_clean()
    core = core_pollutants(missing_pct(get_split(df, "train")))
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    test_rows = (df["split"] == "test").to_numpy()

    row = chosen_row("deep_selection.csv", cell="lstm")
    cfg = {"cell": "lstm", "hidden": int(row["hidden"]), "window": int(row["window"])}
    model, scaler, _ = fit_deep(df, core, cfg, ["train", "val"], int(row["epochs"]))
    app_logic.save_lstm(app_logic.LSTM_PATH, model, scaler, cfg, core, int(row["epochs"]))
    lstm = app_logic.load_lstm()
    check_same("lstm", df, app_logic.predict_lstm(df, lstm, test_rows), "preds_deep.csv", "pred_lstm")

    row = chosen_row("ml_selection.csv", model="xgboost")
    params = ast.literal_eval(row["params"])
    pollutants = pollutants_for(row["feature_set"], core)
    X = build_features(df, pollutants)
    _, model = fit_predict(df, X, "xgboost", params, ["train", "val"], "test")
    app_logic.save_xgboost(model, {"feature_set": row["feature_set"], "params": params,
                                   "pollutants": pollutants, "features": list(X.columns)})
    xgb = app_logic.load_xgboost()
    check_same("xgboost", df, app_logic.predict_xgboost(df, xgb, test_rows), "preds_ml.csv", "pred_xgboost")
    print(f"Models saved to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
