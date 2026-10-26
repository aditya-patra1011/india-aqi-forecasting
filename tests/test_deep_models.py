import numpy as np
import pandas as pd
import pytest

from src import config
from src.deep_models import fit_predict_deep, fit_scaler, make_windows, run_deep_models

N_TRAIN, N_VAL, N_TEST = 200, 50, 30
TINY_GRID = [{"cell": "lstm", "hidden": 8, "window": 5}, {"cell": "lstm", "hidden": 8, "window": 7},
             {"cell": "gru", "hidden": 8, "window": 5}]
COLS = ["PM2.5"]


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


def _windows(df, window=5):
    cols = [*COLS, config.TARGET]
    return make_windows(df, cols, fit_scaler(df, cols, ["train"]), window)


def test_window_for_a_day_never_contains_that_days_own_values():
    df = _frame()
    cols = [*COLS, config.TARGET]
    scaler = fit_scaler(df, cols, ["train"])                 # same scaler for both, to isolate the windows
    seq, static, _ = make_windows(df, cols, scaler, 5)
    k = 50                                                   # row k belongs to city A
    pert = df.copy()
    pert.loc[k, ["AQI", "PM2.5"]] *= 10
    pseq, pstatic, _ = make_windows(pert, cols, scaler, 5)
    assert np.array_equal(seq[: k + 1], pseq[: k + 1])      # windows up to and including day k: unchanged
    assert not np.array_equal(seq[k + 1], pseq[k + 1])       # day k+1 sees yesterday's change
    assert np.array_equal(static, pstatic)                   # static inputs are calendar and city only


def test_window_values_are_the_previous_days_in_order():
    df = _frame()
    cols = [*COLS, config.TARGET]
    mean, std = fit_scaler(df, cols, ["train"])
    seq, _, _ = make_windows(df, cols, (mean, std), 5)
    t = 30
    expected = ((df.loc[t - 5: t - 1, cols] - mean) / std).to_numpy(np.float32)
    assert np.allclose(seq[t, :, :2], expected)              # days t-5 .. t-1, oldest first
    assert np.all(seq[t, :, 2] == 0)                         # AQI-missing flag


def test_windows_do_not_cross_cities_and_need_full_history():
    df = _frame()
    _, _, valid = _windows(df, window=5)
    for city in ("A", "B"):
        pos = np.flatnonzero(df["City"] == city)
        assert not valid[pos[:5]].any()                      # first 5 days have no full window
        assert valid[pos[5:]].all()


def test_missing_values_become_zero_and_aqi_gaps_are_flagged():
    df = _frame()
    df.loc[40, ["AQI", "PM2.5"]] = np.nan
    seq, _, _ = _windows(df, window=5)
    assert np.all(np.isfinite(seq))
    assert np.all(seq[41, -1, :2] == 0) and seq[41, -1, 2] == 1


def test_scaler_uses_only_fit_splits():
    df = _frame()
    cols = [*COLS, config.TARGET]
    pert = df.copy()
    pert.loc[pert["split"] != "train", cols] *= 100
    m1, s1 = fit_scaler(df, cols, ["train"])
    m2, s2 = fit_scaler(pert, cols, ["train"])
    assert m1.equals(m2) and s1.equals(s2)


def test_fit_predict_deep_predicts_only_requested_split_and_honours_fixed_epochs():
    df = _frame()
    pred, n_epochs = fit_predict_deep(df, COLS, TINY_GRID[0], ["train"], "val", epochs=3)
    assert n_epochs == 3
    assert pred[df["split"] == "val"].notna().all()
    assert pred[df["split"] != "val"].isna().all()


def _perturb_test_day(df, position, factor=5.0):
    out = df.copy()
    idx = out.index[(out["City"] == "A") & (out["split"] == "test")][position]
    out.loc[idx, ["AQI", "PM2.5"]] *= factor
    return out


def test_run_deep_models_validation_forecasts_ignore_test_data_and_test_forecasts_ignore_the_future():
    df = _frame()
    base_preds, base_table, _ = run_deep_models(df, COLS, TINY_GRID, max_epochs=3)
    k = 10
    pert_preds, pert_table, _ = run_deep_models(_perturb_test_day(df, k), COLS, TINY_GRID, max_epochs=3)

    val = (df["split"] == "val").to_numpy()
    test_up_to_k = ((df["split"] == "test") & (df["City"] == "A")).to_numpy(copy=True)
    test_up_to_k[np.flatnonzero(test_up_to_k)[k + 1:]] = False        # city A test rows, days 0..k
    for col in base_preds.columns:
        assert np.allclose(base_preds.loc[val, col], pert_preds.loc[val, col])                  # val never sees test
        assert np.allclose(base_preds.loc[test_up_to_k, col], pert_preds.loc[test_up_to_k, col])  # test: no peeking ahead
    assert np.allclose(base_table["val_MAE"], pert_table["val_MAE"])
    assert base_table["epochs"].equals(pert_table["epochs"])


def test_run_deep_models_chooses_lowest_validation_relative_mae_per_cell():
    df = _frame()
    preds, table, chosen = run_deep_models(df, COLS, TINY_GRID, max_epochs=3)
    assert set(chosen) == {"lstm", "gru"}
    assert list(preds.columns) == ["pred_lstm", "pred_gru"]
    for cell, c in chosen.items():
        sub = table[table["cell"] == cell]
        best = sub.loc[sub["val_rel_MAE"].idxmin()]
        assert (c["window"], c["epochs"]) == (best["window"], best["epochs"])
        assert table[(table["cell"] == cell) & table["chosen"]].shape[0] == 1
    assert preds.loc[df["split"] == "test"].notna().all().all()      # test forecasts exist for every row


def test_training_is_reproducible():
    df = _frame()
    a, _ = fit_predict_deep(df, COLS, TINY_GRID[0], ["train"], "val", epochs=2)
    b, _ = fit_predict_deep(df, COLS, TINY_GRID[0], ["train"], "val", epochs=2)
    assert np.allclose(a.dropna(), b.dropna())


def test_unknown_cell_type_is_rejected():
    with pytest.raises(KeyError):
        fit_predict_deep(_frame(), COLS, {"cell": "transformer", "hidden": 8, "window": 5}, ["train"], "val", epochs=1)
