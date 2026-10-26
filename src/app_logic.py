"""Everything the Streamlit app needs that is not UI: saving and loading the final
models, and next-day forecasts from the latest data.

The data ends on 2020-07-01, so the "next day" is 2020-07-02. A forecast row for that
day is appended to each city with all readings unknown (NaN); the models only look at
earlier days, so the forecast uses exactly the information a forecaster would have had
on the evening of 2020-07-01.
"""
import json

import numpy as np
import pandas as pd
import torch

from src import config
from src.cleaning import aqi_to_bucket
from src.comparison import MODELS, add_periods, merge_predictions
from src.deep_models import RecurrentForecaster, predict_deep
from src.features import build_features

LSTM_PATH = config.MODELS_DIR / "lstm.pt"
XGB_PATH = config.MODELS_DIR / "xgboost.json"
XGB_META_PATH = config.MODELS_DIR / "xgboost_meta.json"
FORECAST_SPLIT = "forecast"

# CPCB's own category colors (text on them uses dark ink where the fill is light)
CATEGORY_COLORS = {
    "Good": ("#00b050", "#0b0b0b"), "Satisfactory": ("#92d050", "#0b0b0b"),
    "Moderate": ("#ffd500", "#0b0b0b"), "Poor": ("#ff9900", "#0b0b0b"),
    "Very Poor": ("#e02020", "#ffffff"), "Severe": ("#8b0000", "#ffffff"),
}


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROCESSED / "aqi_clean.csv", parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def prediction_files() -> list:
    return sorted({f for f, _ in MODELS.values() if f is not None})


def missing_artifacts() -> list:
    """Files the app needs that are not on disk, each with the command that makes it."""
    need = {config.DATA_PROCESSED / "aqi_clean.csv": "python -m src.make_dataset",
            config.DATA_PROCESSED / "preds_classical.csv": "python -m src.run_classical",
            config.DATA_PROCESSED / "preds_ml.csv": "python -m src.run_ml",
            config.DATA_PROCESSED / "preds_deep.csv": "python -m src.run_deep",
            LSTM_PATH: "python -m src.export_models", XGB_PATH: "python -m src.export_models"}
    return [(p.relative_to(config.ROOT).as_posix(), cmd) for p, cmd in need.items() if not p.exists()]


def load_history() -> pd.DataFrame:
    """Clean data with every model's saved forecasts, as compared in phase 7."""
    files = {n: pd.read_csv(config.DATA_PROCESSED / n, parse_dates=["Date"]) for n in prediction_files()}
    return add_periods(merge_predictions(load_clean(), files))


def category(aqi: float) -> str:
    """CPCB category of one AQI value (clipped to 0-500)."""
    if aqi is None or np.isnan(aqi):
        return "n/a"
    return aqi_to_bucket(pd.Series([float(np.clip(aqi, 0, 500))])).iloc[0]


def extend_one_day(df: pd.DataFrame) -> pd.DataFrame:
    """Append, for each city, one row for the day after its last date, with every
    reading unknown and split = 'forecast'."""
    new = df.groupby("City", as_index=False)["Date"].max()
    new["Date"] += pd.Timedelta(days=1)
    new["split"] = FORECAST_SPLIT
    out = pd.concat([df, new], ignore_index=True)
    return out.sort_values(["City", "Date"]).reset_index(drop=True)


def apply_overrides(df: pd.DataFrame, city: str, date, values: dict) -> pd.DataFrame:
    """Copy of df with some readings of one city on one date replaced (what-if)."""
    out = df.copy()
    m = (out["City"] == city) & (out["Date"] == pd.Timestamp(date))
    if m.sum() != 1:
        raise ValueError(f"No single row for {city} on {date}")
    for col, v in values.items():
        out.loc[m, col] = v
    return out


def save_lstm(path, model, scaler: tuple, cfg: dict, cols, n_epochs: int) -> None:
    mean, std = scaler
    torch.save({"state_dict": model.state_dict(), "cfg": dict(cfg), "cols": list(cols), "epochs": int(n_epochs),
                "n_seq": model.rnn.input_size, "n_static": model.head[0].in_features - model.rnn.hidden_size,
                "scaler_mean": mean.to_dict(), "scaler_std": std.to_dict()}, path)


def load_lstm(path=LSTM_PATH) -> dict:
    ck = torch.load(path, weights_only=False)
    model = RecurrentForecaster(ck["n_seq"], ck["n_static"], ck["cfg"]["cell"], ck["cfg"]["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    scaler = (pd.Series(ck["scaler_mean"]), pd.Series(ck["scaler_std"]))
    return {"model": model, "scaler": scaler, "cfg": ck["cfg"], "cols": ck["cols"], "epochs": ck["epochs"]}


def save_xgboost(model, meta: dict, path=XGB_PATH, meta_path=XGB_META_PATH) -> None:
    model.save_model(path)
    meta_path.write_text(json.dumps(meta, indent=2))


def load_xgboost(path=XGB_PATH, meta_path=XGB_META_PATH) -> dict:
    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.load_model(path)
    return {"model": model, **json.loads(meta_path.read_text())}


def predict_lstm(df: pd.DataFrame, bundle: dict, rows) -> pd.Series:
    return predict_deep(df, bundle["cols"], bundle["model"], bundle["scaler"], bundle["cfg"]["window"], rows)


def predict_xgboost(df: pd.DataFrame, bundle: dict, rows) -> pd.Series:
    X = build_features(df, bundle["pollutants"])[bundle["features"]]
    rows = np.asarray(rows, bool)
    pred = pd.Series(np.nan, index=df.index)
    pred[rows] = bundle["model"].predict(X[rows])
    return pred


def forecast_next_day(clean: pd.DataFrame, lstm: dict, xgb: dict, overrides=None) -> pd.DataFrame:
    """Next-day forecast per city from persistence, LSTM and XGBoost.

    overrides = (city, {column: value}) replaces readings of that city's LAST observed
    day before forecasting, to show how the forecast reacts (what-if)."""
    df = extend_one_day(clean)
    if overrides is not None:
        city, values = overrides
        last = clean.loc[clean["City"] == city, "Date"].max()
        df = apply_overrides(df, city, last, values)
    rows = (df["split"] == FORECAST_SPLIT).to_numpy()
    out = df.loc[rows, ["City", "Date"]].copy()
    out["persistence"] = df.groupby("City")[config.TARGET].shift(1)[rows]
    out["lstm"] = predict_lstm(df, lstm, rows)[rows]
    out["xgboost"] = predict_xgboost(df, xgb, rows)[rows]
    return out.reset_index(drop=True)
