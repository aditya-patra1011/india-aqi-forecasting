import numpy as np
import pandas as pd
import pytest

from src import app_logic, config
from src.deep_models import fit_deep, predict_deep
from src.features import build_features
from src.ml_models import fit_predict

N_TRAIN, N_VAL, N_TEST = 120, 40, 30
CFG = {"cell": "lstm", "hidden": 8, "window": 5}
COLS = ["PM2.5"]


def _frame(seed=0):
    """Two cities; AQI today depends on PM2.5 yesterday."""
    rng = np.random.default_rng(seed)
    n = N_TRAIN + N_VAL + N_TEST
    split = ["train"] * N_TRAIN + ["val"] * N_VAL + ["test"] * N_TEST
    parts = []
    for c, scale in (("A", 1.0), ("B", 2.0)):
        d = pd.DataFrame({"City": c, "Date": pd.date_range("2019-01-01", periods=n), "split": split})
        pm = rng.uniform(20, 120, n) * scale
        for p in config.POLLUTANTS:
            d[p] = rng.uniform(1, 50, n)
        d["PM2.5"] = pm
        d["AQI"] = 40 + 1.2 * np.r_[pm[0], pm[:-1]] + rng.normal(scale=5, size=n)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    """Tiny LSTM and XGBoost, saved and loaded back like the app does."""
    df = _frame()
    d = tmp_path_factory.mktemp("models")
    model, scaler, n_epochs = fit_deep(df, COLS, CFG, ["train", "val"], epochs=2)
    app_logic.save_lstm(d / "lstm.pt", model, scaler, CFG, COLS, n_epochs)
    X = build_features(df, COLS)
    _, xgb = fit_predict(df, X, "xgboost", {"max_depth": 2, "n_estimators": 20}, ["train", "val"], "test")
    app_logic.save_xgboost(xgb, {"pollutants": COLS, "features": list(X.columns)},
                           d / "xgb.json", d / "xgb_meta.json")
    return {"df": df, "model": model, "scaler": scaler, "xgb": xgb, "X": X,
            "lstm": app_logic.load_lstm(d / "lstm.pt"),
            "xgboost": app_logic.load_xgboost(d / "xgb.json", d / "xgb_meta.json")}


def test_extend_one_day_appends_an_unknown_day_per_city():
    df = _frame()
    out = app_logic.extend_one_day(df)
    new = out[out["split"] == app_logic.FORECAST_SPLIT]
    assert len(out) == len(df) + 2 and set(new["City"]) == {"A", "B"}
    assert (new["Date"] == df["Date"].max() + pd.Timedelta(days=1)).all()
    assert new[[config.TARGET, *config.POLLUTANTS]].isna().all().all()
    build_features(out, COLS)                                   # still one row per consecutive day


def test_apply_overrides_changes_one_cell_and_rejects_unknown_rows():
    df = _frame()
    out = app_logic.apply_overrides(df, "A", df["Date"].max(), {"PM2.5": 999.0})
    changed = (out["PM2.5"] != df["PM2.5"])
    assert changed.sum() == 1 and out.loc[changed, "City"].item() == "A"
    with pytest.raises(ValueError):
        app_logic.apply_overrides(df, "Z", df["Date"].max(), {"PM2.5": 1.0})


def test_saved_lstm_predicts_exactly_like_the_trained_one(bundles):
    df = bundles["df"]
    rows = (df["split"] == "test").to_numpy()
    direct = predict_deep(df, COLS, bundles["model"], bundles["scaler"], CFG["window"], rows)
    loaded = app_logic.predict_lstm(df, bundles["lstm"], rows)
    assert np.allclose(direct[rows], loaded[rows])
    assert bundles["lstm"]["cfg"] == CFG and bundles["lstm"]["cols"] == COLS


def test_saved_xgboost_predicts_exactly_like_the_trained_one(bundles):
    df, X = bundles["df"], bundles["X"]
    rows = (df["split"] == "test").to_numpy()
    loaded = app_logic.predict_xgboost(df, bundles["xgboost"], rows)
    assert np.allclose(bundles["xgb"].predict(X[rows]), loaded[rows], atol=1e-4)


def test_forecast_next_day_gives_one_finite_forecast_per_city(bundles):
    df = bundles["df"]
    out = app_logic.forecast_next_day(df, bundles["lstm"], bundles["xgboost"])
    assert list(out["City"]) == ["A", "B"]
    assert (out["Date"] == df["Date"].max() + pd.Timedelta(days=1)).all()
    assert np.isfinite(out[["persistence", "lstm", "xgboost"]].to_numpy()).all()
    last_aqi = df.groupby("City")[config.TARGET].last()
    assert np.allclose(out.set_index("City")["persistence"], last_aqi)


def test_what_if_changes_only_the_chosen_citys_forecast(bundles):
    df = bundles["df"]
    base = app_logic.forecast_next_day(df, bundles["lstm"], bundles["xgboost"]).set_index("City")
    alt = app_logic.forecast_next_day(df, bundles["lstm"], bundles["xgboost"],
                                      ("A", {"PM2.5": 400.0, "AQI": 300.0})).set_index("City")
    assert alt.loc["A", "persistence"] == 300.0
    assert alt.loc["A", "lstm"] != pytest.approx(base.loc["A", "lstm"])
    assert alt.loc["B"].equals(base.loc["B"])                   # other cities untouched


def test_category_maps_values_and_handles_missing():
    assert app_logic.category(30) == "Good"
    assert app_logic.category(450) == "Severe"
    assert app_logic.category(900) == "Severe"                  # clipped to the 0-500 scale
    assert app_logic.category(float("nan")) == "n/a"
    assert set(app_logic.CATEGORY_COLORS) == {"Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"}


def test_missing_artifacts_lists_paths_with_commands():
    for path, cmd in app_logic.missing_artifacts():
        assert isinstance(path, str) and cmd.startswith("python -m src.")


@pytest.mark.skipif(bool(app_logic.missing_artifacts()), reason="needs the processed data and exported models")
def test_app_runs_end_to_end():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 4
    at.sidebar.selectbox[0].select("Lucknow").run()
    assert not at.exception
    at.slider[0].set_value(300.0).run()                        # what-if PM2.5
    assert not at.exception and len(at.dataframe) >= 1
