import numpy as np
import pandas as pd
import pytest

from src import config
from src.comparison import (FAMILY, MODELS, add_periods, bucket_confusion, common_rows, diebold_mariano,
                            dm_table, dm_test, feature_group, merge_predictions, relative_mae_table,
                            score_table)

DAYS = pd.date_range("2018-01-01", "2020-06-30")      # a full train year, so climatology covers every month


def _clean(seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for c in ("A", "B"):
        d = pd.DataFrame({"City": c, "Date": DAYS})
        d["AQI"] = rng.uniform(50, 300, len(d))
        d["split"] = np.where(d["Date"] <= "2018-12-31", "train",
                              np.where(d["Date"] <= "2019-12-31", "val", "test"))
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _files(clean, noise=None, seed=1):
    """Fake prediction files: every model = truth + its own noise level."""
    rng = np.random.default_rng(seed)
    noise = noise or {}
    files = {}
    for model, (name, col) in MODELS.items():
        if name is None:
            continue
        f = files.setdefault(name, clean[["City", "Date", "AQI"]].copy())
        f[col] = clean["AQI"] + rng.normal(scale=noise.get(model, 20.0), size=len(clean))
    return files


def _merged(noise=None):
    clean = _clean()
    return add_periods(merge_predictions(clean, _files(clean, noise)))


def test_merge_attaches_every_model_and_rejects_stale_files():
    clean = _clean()
    df = merge_predictions(clean, _files(clean))
    assert all(col in df for _, col in MODELS.values())
    stale = _files(clean)
    stale["preds_ml.csv"].loc[5, "AQI"] += 1                  # file built from different data
    with pytest.raises(ValueError, match="preds_ml.csv"):
        merge_predictions(clean, stale)


def test_merge_aligns_on_city_and_date_not_row_order():
    clean = _clean()
    files = _files(clean)
    shuffled = {k: v.sample(frac=1, random_state=0) for k, v in files.items()}
    a = merge_predictions(clean, files)
    b = merge_predictions(clean, shuffled)
    assert np.allclose(a["pred_gru"], b["pred_gru"], equal_nan=True)


def test_periods_split_test_at_lockdown():
    df = _merged()
    test = df[df["split"] == "test"]
    assert set(test["period"]) == {"test_pre_lockdown", "test_lockdown"}
    assert test.loc[test["period"] == "test_lockdown", "Date"].min() == pd.Timestamp("2020-03-25")
    assert (df.loc[df["split"] != "test", "period"] == df.loc[df["split"] != "test", "split"]).all()


def test_all_models_scored_on_the_same_rows():
    df = _merged()
    df.loc[df.index[(df["split"] == "val")][:3], "pred_lstm"] = np.nan     # one model missing 3 days
    scores = score_table(df)
    val_all = scores[(scores["split"] == "val") & (scores["City"] == "ALL")]
    assert val_all["n"].nunique() == 1
    assert val_all["n"].iloc[0] == common_rows(df, {m: c for m, (_, c) in MODELS.items()})[df["split"] == "val"].sum()


def test_skill_of_persistence_is_zero_and_better_models_score_higher():
    df = _merged({"gru": 1.0, "xgboost": 50.0})
    scores = score_table(df)
    assert np.allclose(scores.loc[scores["model"] == "persistence", "skill"], 0.0)
    rel = relative_mae_table(scores, "val")
    assert rel.iloc[0]["model"] == "gru"
    assert set(rel["family"]) == set(FAMILY.values())
    assert set(scores["split"]) == {"val", "test", "test_pre_lockdown", "test_lockdown"}


def test_period_scores_cover_the_test_rows_exactly():
    scores = score_table(_merged())
    s = scores[(scores["City"] == "ALL") & (scores["model"] == "gru")].set_index("split")["n"]
    assert s["test_pre_lockdown"] + s["test_lockdown"] == s["test"]


def test_dm_detects_a_clearly_better_forecast_and_not_an_identical_one():
    rng = np.random.default_rng(0)
    e_good, e_bad = rng.normal(scale=1, size=300), rng.normal(scale=3, size=300)
    r = diebold_mariano(e_good, e_bad)
    assert r["mean_loss_diff"] < 0 and r["p_value"] < 0.01
    same = diebold_mariano(e_good, e_good)
    assert same["mean_loss_diff"] == 0 and np.isnan(same["dm_stat"])


def test_dm_is_not_significant_for_two_equally_good_forecasts():
    rng = np.random.default_rng(1)
    r = diebold_mariano(rng.normal(size=300), rng.normal(size=300))
    assert r["p_value"] > 0.05


def test_dm_test_on_a_loss_difference_keeps_its_sign():
    r = dm_test(np.r_[-np.ones(50), -0.5 * np.ones(50)] + np.random.default_rng(0).normal(scale=0.1, size=100))
    assert r["mean_loss_diff"] < 0 and r["dm_stat"] < 0


def test_dm_table_has_one_row_per_model_and_city_plus_pooled():
    df = _merged({"gru": 1.0})
    out = dm_table(df, ["gru", "xgboost"])
    assert len(out) == 2 * 3
    pooled = out[(out["model"] == "gru") & (out["City"] == "ALL")].iloc[0]
    assert pooled["n"] == df.loc[df["split"] == "test", "Date"].nunique()   # one observation per date
    assert pooled["p_value"] < 0.01


def test_bucket_confusion_keeps_all_categories_and_counts_every_row():
    df = _merged()
    cm = bucket_confusion(df, "pred_gru", "test")
    assert list(cm.index) == list(cm.columns) and len(cm) == 6
    assert cm.to_numpy().sum() == (df["split"] == "test").sum()


def test_feature_groups():
    assert feature_group("aqi_lag1") == "AQI history"
    assert feature_group("PM2.5_lag1") == "pollutant lags"
    assert feature_group("city_Delhi") == "city"
    assert feature_group("doy_sin") == "calendar"
