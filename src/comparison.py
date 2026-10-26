"""Phase 7: one comparison of every model on the same rows.

Every model's forecasts come from the earlier phases (saved prediction files), so
nothing is refitted here except the XGBoost model that SHAP explains, which repeats the
configuration chosen in phase 5. The final model is picked on VALIDATION only; test
numbers are reported once, for information.
"""
import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.baselines import add_baselines
from src.cleaning import BUCKET_LABELS, aqi_to_bucket
from src.evaluation import evaluate

LOCKDOWN_START = "2020-03-25"          # India's national COVID-19 lockdown began on this day

# model name -> (prediction file in data/processed, column); None = recomputed baseline
MODELS = {
    "persistence": (None, "pred_persistence"),
    "mean_7d": (None, "pred_mean_7d"),
    "monthly_climatology": (None, "pred_climatology"),
    "sarimax_fourier": ("preds_classical.csv", "pred_sarimax"),
    "prophet_lag1": ("preds_classical.csv", "pred_prophet_lag1"),
    "ridge": ("preds_ml.csv", "pred_ridge"),
    "random_forest": ("preds_ml.csv", "pred_random_forest"),
    "xgboost": ("preds_ml.csv", "pred_xgboost"),
    "lstm": ("preds_deep.csv", "pred_lstm"),
    "gru": ("preds_deep.csv", "pred_gru"),
}
FAMILY = {"persistence": "baseline", "mean_7d": "baseline", "monthly_climatology": "baseline",
          "sarimax_fourier": "classical", "prophet_lag1": "classical",
          "ridge": "ml", "random_forest": "ml", "xgboost": "ml", "lstm": "deep", "gru": "deep"}
PRED_COLS = {m: col for m, (_, col) in MODELS.items()}


def merge_predictions(clean: pd.DataFrame, files: dict) -> pd.DataFrame:
    """Clean frame + baselines + every saved prediction column, joined on (City, Date).

    `files` maps a file name to its DataFrame. Each file must carry the same AQI values
    as the clean data, which catches prediction files left over from an older dataset."""
    df = add_baselines(clean)
    key = ["City", "Date"]
    for name, pred in files.items():
        cols = [c for (f, c) in MODELS.values() if f == name]
        sub = df[key + [config.TARGET]].merge(pred[key + [config.TARGET] + cols], on=key, how="left",
                                              suffixes=("", "_file"), validate="one_to_one")
        a, b = sub[config.TARGET].to_numpy(float), sub[f"{config.TARGET}_file"].to_numpy(float)
        if not np.allclose(a, b, equal_nan=True):
            raise ValueError(f"{name}: AQI values differ from the clean data; rerun that phase")
        for c in cols:
            df[c] = sub[c].to_numpy()
    return df


def add_periods(df: pd.DataFrame) -> pd.DataFrame:
    """'period' = split, with test divided at the lockdown start."""
    out = df.copy()
    out["period"] = out["split"]
    test = out["split"] == "test"
    out.loc[test, "period"] = np.where(out.loc[test, "Date"] < pd.Timestamp(LOCKDOWN_START),
                                       "test_pre_lockdown", "test_lockdown")
    return out


def common_rows(df: pd.DataFrame, pred_cols: dict) -> pd.Series:
    """Rows where the target and every model's forecast exist, so all models are
    scored on exactly the same days."""
    ok = df[config.TARGET].notna()
    for col in pred_cols.values():
        ok &= df[col].notna()
    return ok


def score_table(df: pd.DataFrame, pred_cols: dict = PRED_COLS, min_true_aqi=None) -> pd.DataFrame:
    """Metrics per period (val, test, test_pre_lockdown, test_lockdown), city and model on
    common rows, plus skill = 1 - MAE / MAE(persistence) for the same period and city."""
    d = df[common_rows(df, pred_cols)]
    parts = []
    for period in ("val", "test", "test_pre_lockdown", "test_lockdown"):
        g = d if period in ("val", "test") else d.assign(split=d["period"])
        parts.append(evaluate(g, pred_cols, period, min_true_aqi=min_true_aqi))
    out = pd.concat(parts, ignore_index=True)
    base = out[out["model"] == "persistence"].set_index(["split", "City"])["MAE"]
    out["skill"] = 1 - out["MAE"] / base.reindex(pd.MultiIndex.from_frame(out[["split", "City"]])).to_numpy()
    out.insert(3, "family", out["model"].map(FAMILY))
    return out


def relative_mae_table(scores: pd.DataFrame, split: str = "val") -> pd.DataFrame:
    """Mean over cities of MAE / MAE(persistence), so every city counts equally. This is
    the same selection rule used inside phases 5 and 6."""
    s = scores[(scores["split"] == split) & (scores["City"] != "ALL")]
    rel = (1 - s["skill"]).groupby(s["model"]).mean().rename(f"{split}_rel_MAE")
    pooled = s.groupby("model")["MAE"].mean().rename(f"{split}_mean_city_MAE")
    out = pd.concat([rel, pooled], axis=1).sort_values(f"{split}_rel_MAE").reset_index()
    out.insert(1, "family", out["model"].map(FAMILY))
    return out


def newey_west_lag(n: int) -> int:
    """Standard automatic bandwidth floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def diebold_mariano(err_a, err_b, max_lag=None) -> dict:
    """Diebold-Mariano test of equal mean absolute error for two forecast error series
    on the same days: d_t = |e_a| - |e_b| (negative mean = A is better)."""
    return dm_test(np.abs(np.asarray(err_a, float)) - np.abs(np.asarray(err_b, float)), max_lag)


def dm_test(loss_diff, max_lag=None) -> dict:
    """DM test that the mean of a loss-difference series is zero. The variance of the
    mean uses Newey-West weights, since daily loss differences are autocorrelated, and
    the two-sided p-value uses a t distribution with n - 1 degrees of freedom and the
    Harvey-Leybourne-Newbold small-sample factor for one-step forecasts."""
    d = np.asarray(loss_diff, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10:
        return {"n": n, "mean_loss_diff": np.nan, "dm_stat": np.nan, "p_value": np.nan}
    lag = newey_west_lag(n) if max_lag is None else max_lag
    u = d - d.mean()
    var = u @ u / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k]) / n
    if var <= 0:
        return {"n": n, "mean_loss_diff": float(d.mean()), "dm_stat": np.nan, "p_value": np.nan}
    dm = d.mean() / np.sqrt(var / n) * np.sqrt((n - 1) / n)
    return {"n": n, "mean_loss_diff": float(d.mean()), "dm_stat": float(dm),
            "p_value": float(2 * stats.t.sf(abs(dm), df=n - 1))}


def dm_table(df: pd.DataFrame, models, ref: str = "persistence", split: str = "test",
             pred_cols: dict = PRED_COLS) -> pd.DataFrame:
    """DM test of each model against `ref`, per city and pooled. The pooled test averages
    the loss difference over cities for each date first, so the five cities' errors on
    the same day (which are correlated) count as one observation."""
    d = df[common_rows(df, pred_cols) & (df["split"] == split)]
    rows = []
    for m in models:
        e_m = d[pred_cols[m]] - d[config.TARGET]
        e_r = d[pred_cols[ref]] - d[config.TARGET]
        for city in sorted(d["City"].unique()):
            c = (d["City"] == city).to_numpy()
            rows.append({"model": m, "ref": ref, "split": split, "City": city,
                         **diebold_mariano(e_m[c], e_r[c])})
        daily = (e_m.abs() - e_r.abs()).groupby(d["Date"]).mean()
        rows.append({"model": m, "ref": ref, "split": split, "City": "ALL",
                     **dm_test(daily.to_numpy())})
    return pd.DataFrame(rows)


def bucket_confusion(df: pd.DataFrame, pred_col: str, split: str = "test") -> pd.DataFrame:
    """Counts of true CPCB category (rows) vs predicted category (columns)."""
    d = df[(df["split"] == split) & df[config.TARGET].notna() & df[pred_col].notna()]
    true = pd.Categorical(aqi_to_bucket(d[config.TARGET]), categories=BUCKET_LABELS)
    pred = pd.Categorical(aqi_to_bucket(d[pred_col].clip(0, 500)), categories=BUCKET_LABELS)
    return pd.crosstab(pd.Series(true, name="true"), pd.Series(pred, name="predicted"), dropna=False)


def feature_group(name: str) -> str:
    """Which kind of information a phase-5 feature carries."""
    if name.startswith("aqi_"):
        return "AQI history"
    if name.startswith("city_"):
        return "city"
    if name in ("doy_sin", "doy_cos", "month"):
        return "calendar"
    return "pollutant lags"


def xgboost_shap(df: pd.DataFrame, core, chosen: dict, split: str = "test") -> pd.DataFrame:
    """Mean |SHAP| per feature for the phase-5 XGBoost configuration, refitted exactly as
    in phase 5 (train for validation rows, train + validation for test rows)."""
    import shap

    from src.features import build_features, pollutants_for
    from src.ml_models import fit_predict

    X = build_features(df, pollutants_for(chosen["feature_set"], core))
    train_splits = ["train"] if split == "val" else ["train", "val"]
    _, model = fit_predict(df, X, "xgboost", chosen["params"], train_splits, split)
    rows = (df["split"] == split).to_numpy()
    values = shap.TreeExplainer(model).shap_values(X[rows])
    out = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(values).mean(axis=0)})
    out["group"] = out["feature"].map(feature_group)
    out["share"] = out["mean_abs_shap"] / out["mean_abs_shap"].sum()
    return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
