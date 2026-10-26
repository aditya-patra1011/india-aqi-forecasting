import numpy as np
import pandas as pd
import pytest

from src import config
from src.ml_models import fit_predict, fit_predict_per_city, pooled_vs_per_city, relative_mae, run_models
from src.features import build_features

N_TRAIN, N_VAL, N_TEST = 300, 60, 40
TINY_GRIDS = {
    "xgboost": [{"max_depth": 2, "n_estimators": 20}, {"max_depth": 3, "n_estimators": 20}],
    "random_forest": [{"min_samples_leaf": 5}],
    "ridge": [{}],
}
FS = ("A_aqi_only", "B_core")


def _frame(seed=0):
    """Two cities; AQI today depends on PM2.5 yesterday, so the models have signal."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL + N_TEST
    split = ["train"] * N_TRAIN + ["val"] * N_VAL + ["test"] * N_TEST
    parts = []
    for c, scale in (("A", 1.0), ("B", 2.0)):
        d = pd.DataFrame({"City": c, "Date": pd.date_range("2018-01-01", periods=n), "split": split})
        pm = rng.uniform(20, 120, n) * scale
        for p in config.POLLUTANTS:
            d[p] = rng.uniform(1, 50, n)
        d["PM2.5"] = pm
        d["AQI"] = 40 + 1.2 * np.r_[pm[0], pm[:-1]] + rng.normal(scale=5, size=n)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _perturb_test_day(df, position, factor=5.0):
    out = df.copy()
    idx = out.index[(out["City"] == "A") & (out["split"] == "test")][position]
    out.loc[idx, ["AQI", "PM2.5"]] *= factor
    return out


def test_relative_mae_of_persistence_itself_is_one():
    df = _frame()
    persist = df.groupby("City")["AQI"].shift(1)
    assert relative_mae(df, persist, "val") == pytest.approx(1.0)


def test_fit_predict_ignores_rows_with_missing_target_and_predicts_only_requested_split():
    df = _frame()
    df.loc[df.index[:10], "AQI"] = np.nan                    # missing targets inside train
    X = build_features(df, ["PM2.5"])
    pred, _ = fit_predict(df, X, "ridge", {}, ["train"], "val")
    assert pred[df["split"] == "val"].notna().all()
    assert pred[df["split"] != "val"].isna().all()


def test_run_models_validation_forecasts_ignore_test_data_and_test_forecasts_ignore_the_future():
    df = _frame()
    base_preds, base_table, _, _ = run_models(df, ["PM2.5"], TINY_GRIDS, FS)
    k = 10
    pert_preds, pert_table, _, _ = run_models(_perturb_test_day(df, k), ["PM2.5"], TINY_GRIDS, FS)

    val = (df["split"] == "val").to_numpy()
    test_up_to_k = ((df["split"] == "test") & (df["City"] == "A")).to_numpy(copy=True)
    test_up_to_k[np.flatnonzero(test_up_to_k)[k + 1:]] = False        # city A test rows, days 0..k
    for col in base_preds.columns:
        assert np.allclose(base_preds.loc[val, col], pert_preds.loc[val, col])                  # val never sees test
        assert np.allclose(base_preds.loc[test_up_to_k, col], pert_preds.loc[test_up_to_k, col])  # test: no peeking ahead
    assert base_table[["model", "feature_set"]].equals(pert_table[["model", "feature_set"]])
    assert np.allclose(base_table["val_MAE"], pert_table["val_MAE"])   # float noise only (~1e-14)


def test_run_models_chooses_lowest_validation_relative_mae_and_learns_the_signal():
    df = _frame()
    preds, table, chosen, _ = run_models(df, ["PM2.5"], TINY_GRIDS, FS)
    for kind in TINY_GRIDS:
        sub = table[table["model"] == kind]
        best = sub.loc[sub["val_rel_MAE"].idxmin()]
        assert chosen[kind]["feature_set"] == best["feature_set"]
        assert table[(table["model"] == kind) & table["chosen"]].shape[0] == 1
    assert chosen["ridge"]["feature_set"] == "B_core"                # yesterday's PM2.5 carries the signal
    assert relative_mae(pd.concat([df, preds], axis=1), preds["pred_ridge"], "val") < 0.5


def test_per_city_fit_for_one_city_ignores_other_cities_data():
    df = _frame()
    X = build_features(df, ["PM2.5"])
    base = fit_predict_per_city(df, X, "ridge", {}, ["train"], "val")
    pert = df.copy()
    pert.loc[pert["City"] == "B", ["AQI", "PM2.5"]] *= 3.0
    pred = fit_predict_per_city(pert, build_features(pert, ["PM2.5"]), "ridge", {}, ["train"], "val")
    a_val = ((df["City"] == "A") & (df["split"] == "val")).to_numpy()
    assert np.allclose(base[a_val], pred[a_val])
    assert pred[df["split"] != "val"].isna().all()


def test_pooled_vs_per_city_reports_one_row_per_model():
    df = _frame()
    chosen = {"ridge": {"feature_set": "B_core", "params": {}}}
    out = pooled_vs_per_city(df, ["PM2.5"], chosen)
    assert list(out["model"]) == ["ridge"]
    assert out[["pooled_val_rel_MAE", "per_city_val_rel_MAE"]].notna().all().all()
