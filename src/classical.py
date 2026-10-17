"""Classical next-day AQI forecasters: SARIMAX with Fourier seasonality, and Prophet
with yesterday's AQI as a regressor.

Both models forecast day d from information up to day d - 1:
  * SARIMAX parameters are fitted once, then the model is rolled forward day by day
    with fixed parameters (statsmodels `extend`), so each forecast only uses the
    filtered state up to the previous day.
  * Prophet is fitted once, and its lag-1 regressor for day d is the actual AQI of day
    d - 1, which is known when the forecast is made.
Validation forecasts come from models fitted on train only. Test forecasts come from
models refitted on train + validation, with settings chosen on validation only.
"""
import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src import config
from src.evaluation import regression_metrics

SARIMAX_ORDERS = [(1, 0, 0), (2, 0, 0), (3, 0, 0), (1, 0, 1), (2, 0, 1)]
FOURIER_KS = [2, 4]


def fourier_terms(dates, k: int) -> pd.DataFrame:
    """Sine/cosine pairs for yearly seasonality. They depend only on the calendar date,
    so they are known in advance and cannot leak anything."""
    idx = pd.DatetimeIndex(dates)
    t = (idx - pd.Timestamp("2000-01-01")).days.to_numpy()
    cols = {}
    for j in range(1, k + 1):
        cols[f"sin{j}"] = np.sin(2 * np.pi * j * t / 365.25)
        cols[f"cos{j}"] = np.cos(2 * np.pi * j * t / 365.25)
    return pd.DataFrame(cols, index=idx)


def city_series(df: pd.DataFrame, city: str, splits) -> pd.Series:
    """Daily AQI series of one city for the given split names."""
    g = df[(df["City"] == city) & (df["split"].isin(splits))].sort_values("Date")
    return g.set_index("Date")[config.TARGET].asfreq("D")


def fit_sarimax(y: pd.Series, order: tuple, k: int):
    """SARIMAX on log(AQI) with a constant, Fourier seasonality and ARMA errors."""
    if not (y.dropna() > 0).all():
        raise ValueError("AQI must be positive to take logs")
    model = SARIMAX(np.log(y), exog=fourier_terms(y.index, k), order=order, trend="c")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False, maxiter=200)


def one_step_forecasts(res, y_new: pd.Series, k: int) -> pd.Series:
    """One-step-ahead forecasts for y_new using fixed parameters. The forecast for a
    day uses data only up to the day before. Returned on the AQI scale."""
    ext = res.extend(np.log(y_new), exog=fourier_terms(y_new.index, k))
    return np.exp(ext.fittedvalues)


def select_sarimax(y_train: pd.Series, y_val: pd.Series, orders=SARIMAX_ORDERS, ks=FOURIER_KS):
    """Fit every (order, K) on train, score on validation, and return the best
    converged candidate together with the full candidate table."""
    rows = []
    for order in orders:
        for k in ks:
            res = fit_sarimax(y_train, order, k)
            pred = one_step_forecasts(res, y_val, k)
            m = regression_metrics(y_val, pred)
            rows.append({"order": str(order), "K": k, "val_MAE": m["MAE"],
                         "converged": bool(res.mle_retvals.get("converged", False))})
    table = pd.DataFrame(rows)
    ok = table[table["converged"]]
    if ok.empty:
        raise RuntimeError("No SARIMAX candidate converged")
    best = ok.loc[ok["val_MAE"].idxmin()]
    best_order = tuple(int(c) for c in best["order"].strip("()").split(",") if c.strip())
    return best_order, int(best["K"]), table


def sarimax_city_forecasts(df: pd.DataFrame, city: str, orders=SARIMAX_ORDERS, ks=FOURIER_KS):
    """Validation + test forecasts for one city, and the selection table."""
    train = city_series(df, city, ["train"])
    val = city_series(df, city, ["val"])
    test = city_series(df, city, ["test"])

    order, k, table = select_sarimax(train, val, orders, ks)
    pred_val = one_step_forecasts(fit_sarimax(train, order, k), val, k)

    train_val = pd.concat([train, val])                    # refit on train + validation only
    pred_test = one_step_forecasts(fit_sarimax(train_val, order, k), test, k)

    table.insert(0, "City", city)
    table["chosen"] = (table["order"] == str(order)) & (table["K"] == k)
    return pd.concat([pred_val, pred_test]).rename("pred_sarimax"), table


def _quiet_prophet() -> None:
    for name in ("cmdstanpy", "prophet"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _fit_prophet(frame: pd.DataFrame):
    from prophet import Prophet  # imported lazily so the rest of the module needs only statsmodels
    _quiet_prophet()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                uncertainty_samples=0)
    m.add_regressor("lag1")
    m.fit(frame.dropna(subset=["y", "lag1"])[["ds", "y", "lag1"]])
    return m


def _predict_prophet(model, frame: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=frame["ds"].to_numpy())
    ok = frame["lag1"].notna().to_numpy()
    if ok.any():
        yhat = model.predict(frame.loc[ok, ["ds", "lag1"]])["yhat"].to_numpy()
        out.iloc[np.flatnonzero(ok)] = yhat
    return out


def prophet_city_forecasts(df: pd.DataFrame, city: str) -> pd.Series:
    """Prophet with yearly seasonality and yesterday's AQI as a regressor. Settings are
    fixed in advance (nothing is tuned). Validation forecasts come from a model fitted
    on train; test forecasts from a model refitted on train + validation."""
    g = df[df["City"] == city].sort_values("Date")
    frame = pd.DataFrame({
        "ds": g["Date"].to_numpy(),
        "y": g[config.TARGET].to_numpy(),
        "lag1": g[config.TARGET].shift(1).to_numpy(),      # AQI of the previous day
        "split": g["split"].to_numpy(),
    })
    train = frame[frame["split"] == "train"]
    val = frame[frame["split"] == "val"]
    test = frame[frame["split"] == "test"]

    pred_val = _predict_prophet(_fit_prophet(train), val)
    pred_test = _predict_prophet(_fit_prophet(pd.concat([train, val])), test)
    return pd.concat([pred_val, pred_test]).rename("pred_prophet_lag1")