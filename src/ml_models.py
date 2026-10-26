"""Machine-learning forecasters for next-day AQI: XGBoost, Random Forest and Ridge.

Protocol (same as the classical models):
  * every configuration (feature set x hyperparameters) is fitted on TRAIN rows only
    and scored on VALIDATION;
  * the best configuration per model is chosen on validation only;
  * validation forecasts come from the train-only fit, test forecasts from a refit on
    train + validation. Test data is never used for any choice.
One pooled model is fitted across cities (city is a feature). `pooled_vs_per_city`
checks that choice on validation against one model per city, using each model's
chosen configuration (reports/tables/ml_pooled_vs_per_city.csv): pooling is better
for XGBoost and Random Forest, while per-city Ridge is marginally better (relative
MAE 0.926 vs 0.934). Pooled is kept for all three so they share one setup.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src import config
from src.features import FEATURE_SETS, build_features, pollutants_for

GRIDS = {
    "xgboost": [{"max_depth": d, "n_estimators": n} for d in (3, 5) for n in (200, 500)],
    "random_forest": [{"min_samples_leaf": leaf} for leaf in (3, 10)],
    "ridge": [{}],
}


def make_model(kind: str, **params):
    if kind == "xgboost":
        return XGBRegressor(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            random_state=config.RANDOM_STATE, n_jobs=1, verbosity=0, **params)
    if kind == "random_forest":
        return RandomForestRegressor(n_estimators=300, max_features=0.5,
                                     random_state=config.RANDOM_STATE, n_jobs=-1, **params)
    if kind == "ridge":
        # imputer and scaler live inside the pipeline, so they are fitted on training rows only
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), Ridge(alpha=1.0))
    raise ValueError(f"Unknown model: {kind}")


def fit_predict(df: pd.DataFrame, X: pd.DataFrame, kind: str, params: dict,
                train_splits, pred_split: str):
    """Fit on rows of the given splits (target must be known) and predict one split."""
    train_rows = df["split"].isin(train_splits) & df[config.TARGET].notna()
    pred_rows = df["split"] == pred_split
    model = make_model(kind, **params)
    model.fit(X[train_rows], df.loc[train_rows, config.TARGET])
    pred = pd.Series(np.nan, index=df.index)
    pred[pred_rows] = model.predict(X[pred_rows])
    return pred, model


def fit_predict_per_city(df: pd.DataFrame, X: pd.DataFrame, kind: str, params: dict,
                         train_splits, pred_split: str) -> pd.Series:
    """Same as fit_predict, but one model per city, fitted on that city's rows only."""
    pred = pd.Series(np.nan, index=df.index)
    for city in df["City"].unique():
        m = df["City"] == city
        pred[m], _ = fit_predict(df[m], X[m], kind, params, train_splits, pred_split)
    return pred


def pooled_mae(df: pd.DataFrame, pred: pd.Series, split: str) -> float:
    ok = (df["split"] == split) & pred.notna() & df[config.TARGET].notna()
    return float((pred[ok] - df.loc[ok, config.TARGET]).abs().mean())


def relative_mae(df: pd.DataFrame, pred: pd.Series, split: str = "val") -> float:
    """Average over cities of MAE(model) / MAE(persistence). Below 1 beats persistence,
    and every city counts equally (pooled MAE is dominated by Delhi and Lucknow)."""
    persist = df.groupby("City")[config.TARGET].shift(1)
    ok = (df["split"] == split) & pred.notna() & persist.notna() & df[config.TARGET].notna()
    city = df.loc[ok, "City"]
    err_model = (pred[ok] - df.loc[ok, config.TARGET]).abs().groupby(city).mean()
    err_base = (persist[ok] - df.loc[ok, config.TARGET]).abs().groupby(city).mean()
    return float((err_model / err_base).mean())


def run_models(df: pd.DataFrame, core, grids=GRIDS, feature_sets=FEATURE_SETS):
    """Select on validation, then produce validation and test forecasts.

    Returns (predictions DataFrame with one pred_<model> column, selection table,
    chosen configs, final models fitted on train + validation, feature names per set)."""
    feats = {fs: build_features(df, pollutants_for(fs, core)) for fs in feature_sets}

    rows, candidates = [], []
    for kind, grid in grids.items():
        for fs in feature_sets:
            for params in grid:
                pred_val, _ = fit_predict(df, feats[fs], kind, params, ["train"], "val")
                rel = relative_mae(df, pred_val)
                rows.append({"model": kind, "feature_set": fs, "params": str(params),
                             "val_MAE": pooled_mae(df, pred_val, "val"), "val_rel_MAE": rel})
                candidates.append((kind, fs, params, rel))
    table = pd.DataFrame(rows)

    preds = pd.DataFrame(index=df.index)
    chosen, final_models = {}, {}
    for kind in grids:
        _, fs, params, _ = min((c for c in candidates if c[0] == kind), key=lambda c: c[3])
        pred_val, _ = fit_predict(df, feats[fs], kind, params, ["train"], "val")
        pred_test, model = fit_predict(df, feats[fs], kind, params, ["train", "val"], "test")
        preds[f"pred_{kind}"] = pred_val.combine_first(pred_test)
        chosen[kind] = {"feature_set": fs, "params": params}
        final_models[kind] = (model, list(feats[fs].columns))
    table["chosen"] = [
        chosen[r["model"]]["feature_set"] == r["feature_set"] and str(chosen[r["model"]]["params"]) == r["params"]
        for r in rows]
    return preds, table, chosen, final_models


def pooled_vs_per_city(df: pd.DataFrame, core, chosen: dict) -> pd.DataFrame:
    """Validation scores of one pooled model vs one model per city, each with the
    configuration already chosen for the pooled model. Train-only fits, so test is
    never touched."""
    rows = []
    for kind, c in chosen.items():
        X = build_features(df, pollutants_for(c["feature_set"], core))
        pooled, _ = fit_predict(df, X, kind, c["params"], ["train"], "val")
        per_city = fit_predict_per_city(df, X, kind, c["params"], ["train"], "val")
        rows.append({"model": kind, "feature_set": c["feature_set"], "params": str(c["params"]),
                     "pooled_val_MAE": pooled_mae(df, pooled, "val"),
                     "per_city_val_MAE": pooled_mae(df, per_city, "val"),
                     "pooled_val_rel_MAE": relative_mae(df, pooled),
                     "per_city_val_rel_MAE": relative_mae(df, per_city)})
    return pd.DataFrame(rows)