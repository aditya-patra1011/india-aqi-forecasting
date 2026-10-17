"""
Metrics and scoring helpers shared by every model in the project.
"""
import pandas as pd
import numpy as np

from src import config
from src.cleaning import aqi_to_bucket

def bucket_accuracy(y_true, y_pred) -> float:
    """Share of days where the predicted CPCB category equals the true one.
    Predictions are clipped to the valid 0-500 AQI range first."""
    t = aqi_to_bucket(pd.Series(np.asarray(y_true, dtype=float)))
    p = aqi_to_bucket(pd.Series(np.clip(np.asarray(y_pred, dtype=float), 0, 500)))
    ok = t.notna() & p.notna()
    return float((t[ok] == p[ok]).mean()) if ok.any() else float("nan")

def regression_metrics(y_true, y_pred) -> dict:
    """n, RMSE, MAPE (%) and category accuracy. Rows where either value is NaN
    are dropped, and n reports how many rows were actually scored"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[ok], y_pred[ok]
    if len(yt) == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "bucket_acc": np.nan}
    err = yp - yt
    nz = yt != 0
    mape = float(np.mean(np.abs(err[nz] / yt[nz])) * 100) if nz.any() else np.nan
    return {
        "n": int(len(yt)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": mape,
        "bucket_acc": bucket_accuracy(yt, yp),
    }

def evaluate(df: pd.DataFrame, pred_cols: dict, split: str,
            target: str = config.TARGET, min_true_aqi=None) -> pd.DataFrame:
    """
    Score each model on one split, per city and pooled ('ALL').

    pred_cols maps a model name to its prediction column in df. Rows are selected by
    the date of the value being predicted. min_true_aqi optionally keeps only days
    whose actual AQI is at least that value, as a diagnostic of how models behave
    on severe-pollution days (not a forecast metric, since it conditions on the outcome).
    """
    d = df[df["split"] == split]
    if min_true_aqi is not None:
        d = d[d[target] >= min_true_aqi]
    rows = []
    for model, col in pred_cols.items():
        for city, g in [*d.groupby("City"), ("ALL", d)]:
            rows.append({"split": split, "City": city, "model": model,
                         **regression_metrics(g[target], g[col])})
    return pd.DataFrame(rows)