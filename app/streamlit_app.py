"""India AQI forecasting app. Run from the repo root:
    streamlit run app/streamlit_app.py
"""
import calendar
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import app_charts as charts
from src import app_logic, config
from src.comparison import FAMILY, LOCKDOWN_START, PRED_COLS
from src.evaluation import regression_metrics

TABLES = ROOT / "reports" / "tables"
HISTORY_MODELS = ["persistence", "xgboost", "lstm", "random_forest", "gru", "sarimax_fourier"]

st.set_page_config(page_title="India AQI Forecasting", page_icon="🌫️", layout="wide")


@st.cache_data
def load_data():
    return app_logic.load_clean(), app_logic.load_history()


@st.cache_resource
def load_models():
    return app_logic.load_lstm(), app_logic.load_xgboost()


@st.cache_data
def next_day(_clean, _lstm, _xgb, overrides=None):
    return app_logic.forecast_next_day(_clean, _lstm, _xgb, overrides)


@st.cache_data
def table(name: str, **kw) -> pd.DataFrame:
    return charts.read(name, **kw)


def chart(fig, data: pd.DataFrame = None, note: str = "") -> None:
    """A chart, an optional one-line takeaway, and its numbers behind an expander."""
    st.plotly_chart(fig, width="stretch")
    if note:
        st.markdown(note)
    if data is not None:
        with st.expander("Show data"):
            st.dataframe(data.round(3), hide_index=True, width="stretch")


def badge(aqi: float) -> str:
    cat = app_logic.category(aqi)
    bg, fg = app_logic.CATEGORY_COLORS.get(cat, ("#e4e3df", "#0b0b0b"))
    return (f"<span style='background:{bg};color:{fg};padding:2px 10px;border-radius:12px;"
            f"font-size:0.85rem;font-weight:600'>{cat}</span>")


def forecast_card(col, title: str, value: float, note: str) -> None:
    col.markdown(f"**{title}**")
    col.markdown(f"<div style='font-size:2.4rem;font-weight:700;line-height:1.1'>{value:.0f}</div>",
                 unsafe_allow_html=True)
    col.markdown(badge(value), unsafe_allow_html=True)
    col.caption(note)


missing = app_logic.missing_artifacts()
if missing:
    st.title("India AQI Forecasting")
    st.error("Some files the app needs are missing. Run these from the repo root first:")
    st.code("\n".join(dict.fromkeys(cmd for _, cmd in missing)), language="bash")
    st.caption("Missing: " + ", ".join(p for p, _ in missing))
    st.stop()

clean, history = load_data()
lstm, xgb = load_models()

st.title("India AQI Forecasting")
st.caption("Next-day AQI for five Indian cities, 2015-2020 (CPCB data). Every model was chosen on "
           "2019 validation data only; 2020 is held-out test data.")
city = st.sidebar.selectbox("City", config.CITIES)
st.sidebar.markdown("**Final model:** LSTM, picked on validation from 10 models (phase 7).")
st.sidebar.caption(f"Data ends {clean['Date'].max():%d %b %Y}.")

dark = st.context.theme.type == "dark"
tab_fc, tab_hist, tab_eda, tab_models, tab_board, tab_about = st.tabs(
    ["Next-day forecast", "History", "Data & EDA", "Model analysis", "Leaderboard", "About"])

with tab_fc:
    base = next_day(clean, lstm, xgb)
    row = base[base["City"] == city].iloc[0]
    last = clean[clean["City"] == city].iloc[-1]
    st.subheader(f"{city}: forecast for {row['Date']:%a %d %b %Y}")
    st.caption(f"Based on data up to {last['Date']:%d %b %Y}, when AQI was {last['AQI']:.0f} "
               f"({app_logic.category(last['AQI'])}).")
    c1, c2, c3 = st.columns(3)
    forecast_card(c1, "LSTM (final model)", row["lstm"], "Recurrent network on the last 14 days")
    forecast_card(c2, "XGBoost", row["xgboost"], "Gradient boosting on lag features")
    forecast_card(c3, "Persistence", row["persistence"], "Tomorrow = today")

    st.divider()
    st.markdown("#### What if yesterday had been different?")
    st.caption(f"Change {city}'s readings for {last['Date']:%d %b %Y} and see how the models react. "
               "Persistence simply follows the AQI you set.")
    w1, w2 = st.columns(2)
    pm = w1.slider("PM2.5 (µg/m³)", 0.0, 500.0, float(round(last["PM2.5"], 1)), 1.0)
    aqi = w2.slider("AQI", 0.0, 500.0, float(round(last["AQI"], 0)), 1.0)
    if pm != round(last["PM2.5"], 1) or aqi != round(last["AQI"], 0):
        alt = next_day(clean, lstm, xgb, (city, {"PM2.5": pm, "AQI": aqi}))
        arow = alt[alt["City"] == city].iloc[0]
        whatif = pd.DataFrame({"model": ["LSTM", "XGBoost", "Persistence"],
                               "actual data": [row["lstm"], row["xgboost"], row["persistence"]],
                               "what-if": [arow["lstm"], arow["xgboost"], arow["persistence"]]})
        whatif["change"] = whatif["what-if"] - whatif["actual data"]
        whatif["category (what-if)"] = whatif["what-if"].map(app_logic.category)
        st.dataframe(whatif.round(1), hide_index=True, width="stretch")
    else:
        st.info("Move a slider to compare.")

with tab_hist:
    h = history[(history["City"] == city) & history["split"].isin(["val", "test"])]
    lo, hi = h["Date"].min().date(), h["Date"].max().date()
    d1, d2 = st.slider("Date range", lo, hi, (pd.Timestamp("2020-01-01").date(), hi), format="DD MMM YYYY")
    chosen = st.multiselect("Models", HISTORY_MODELS, default=["persistence", "xgboost", "lstm"],
                            format_func=charts.label)
    h = h[(h["Date"].dt.date >= d1) & (h["Date"].dt.date <= d2)]
    ink, ink_2 = ("#f0efec", "#c3c2b7") if dark else ("#0b0b0b", "#52514e")
    colors = charts.model_colors(dark)

    fig = go.Figure()
    fig.add_scatter(x=h["Date"], y=h[config.TARGET], name="Actual", line=dict(color=ink, width=2.5))
    for m in chosen:
        dash = "dash" if m == "persistence" else "solid"
        fig.add_scatter(x=h["Date"], y=h[PRED_COLS[m]], name=charts.label(m),
                        line=dict(color=colors[m], width=2, dash=dash))
    if d1 <= pd.Timestamp(LOCKDOWN_START).date() <= d2:
        fig.add_vline(x=pd.Timestamp(LOCKDOWN_START), line=dict(color=ink_2, width=1))
        fig.add_annotation(x=pd.Timestamp(LOCKDOWN_START), y=1, yref="paper", text="lockdown starts",
                           showarrow=False, xanchor="left", font=dict(color=ink_2, size=11))
    fig.update_layout(hovermode="x unified", height=450, margin=dict(l=10, r=10, t=70, b=10),
                      yaxis_title="AQI", legend=dict(orientation="h", y=1.2, x=0),
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig, width="stretch")

    if chosen:
        rows = [{"model": charts.label(m), **regression_metrics(h[config.TARGET], h[PRED_COLS[m]])} for m in chosen]
        err = pd.DataFrame(rows).rename(columns={"bucket_acc": "category accuracy", "MAPE": "MAPE %"})
        st.caption(f"Errors over the selected {len(h)} days")
        st.dataframe(err.round(2), hide_index=True, width="stretch")

with tab_eda:
    st.markdown("All analysis here uses the **train split (2015-2018) only**, as in phase 2, so it never "
                "informs a choice with validation or test data.")
    n_days = clean.groupby("City").size()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cities", len(n_days))
    k2.metric("Days per city", f"{n_days.min():,}-{n_days.max():,}")
    k3.metric("Pollutants", len(config.POLLUTANTS))
    k4.metric("AQI values capped at 500", int(clean["AQI_capped"].sum()))

    eda_cities = st.multiselect("Cities shown in the line charts", config.CITIES, default=config.CITIES,
                                key="eda_cities", help="Legend clicks also hide or show a city.")
    st.caption("Hover for exact values, drag to zoom, double-click to reset.")

    st.markdown("#### Seasonality")
    by = st.radio("Group by", ["month", "weekday", "year"], horizontal=True, key="profile_by")
    prof = table({"month": "monthly_profile", "weekday": "weekday_profile", "year": "yearly_profile"}[by])
    peak = table("monthly_profile").set_index("month")["Delhi"]
    chart(charts.show_only(charts.profile_chart(prof, by, dark), eda_cities), prof,
          f"Winter drives everything in the north: Delhi averages **{peak.max():.0f}** in "
          f"{calendar.month_name[int(peak.idxmax())]} vs **{peak.min():.0f}** in "
          f"{calendar.month_name[int(peak.idxmin())]}. Bengaluru barely moves through the year.")

    st.markdown("#### How bad does it get?")
    chart(charts.category_share_chart(clean), note="Delhi and Lucknow spend a large share of days in "
          "Poor or worse; the southern cities are mostly Satisfactory or Moderate.")

    st.markdown("#### Persistence is hard to beat")
    ac = table("autocorr")
    lag1 = ac.set_index(ac.columns[0]).loc[1]
    chart(charts.show_only(charts.autocorr_chart(ac, dark), eda_cities),
          ac.rename(columns={ac.columns[0]: "lag (days)"}),
          f"Yesterday's AQI explains a lot of today's: lag-1 autocorrelation ranges from "
          f"**{lag1.min():.2f}** ({lag1.idxmin()}) to **{lag1.max():.2f}** ({lag1.idxmax()}). That is why "
          "\"tomorrow = today\" is the baseline every model must beat.")

    st.markdown("#### Which pollutants carry next-day signal?")
    lc = table("lagged_corr_lag1")
    chart(charts.lagged_corr_heatmap(lc, dark), lc.rename(columns={lc.columns[0]: "pollutant"}),
          "Yesterday's PM2.5 is the strongest single predictor in every city. Gaps in the map are "
          "pollutants that are never measured there.")

    st.markdown("#### Data quality")
    ms = table("missing_pct")
    chart(charts.missing_heatmap(ms, dark), ms.rename(columns={ms.columns[0]: "column"}),
          "PM10 and Xylene are missing entirely in some cities, so the 'core' feature set keeps only "
          "pollutants that are at most 25% missing in **every** city (rule fixed on train).")

with tab_models:
    ph_base, ph_classical, ph_ml, ph_deep, ph_cmp = st.tabs(
        ["Baselines", "Classical", "Machine learning", "Deep learning", "Comparison"])
    split_name = {"val": "validation 2019", "test": "test H1 2020"}

    with ph_base:
        b = table("baseline_results")
        sp = st.radio("Split", ["val", "test"], horizontal=True, key="base_split", format_func=split_name.get)
        bmodels = ["persistence", "mean_7d", "seasonal_naive_7d", "seasonal_naive_365d", "monthly_climatology"]
        chart(charts.city_mae_chart(b, sp, bmodels, dark, f"Baseline MAE by city ({split_name[sp]})"),
              b[b["split"] == sp].pivot(index="City", columns="model", values="MAE").reset_index(),
              "Persistence wins in every city. Averaging the last week, or using last year's value, "
              "throws away the strong day-to-day memory.")

    with ph_classical:
        c = table("classical_results")
        sp = st.radio("Split", ["val", "test"], horizontal=True, key="cls_split", format_func=split_name.get)
        chart(charts.city_mae_chart(c, sp, ["persistence", "sarimax_fourier", "prophet_lag1"], dark,
                                    f"SARIMAX and Prophet vs persistence ({split_name[sp]})"),
              c[c["split"] == sp].pivot(index="City", columns="model", values="MAE").reset_index(),
              "SARIMAX is only a few percent better than persistence on validation and worse on test; "
              "Prophet is worse than persistence. The series is dominated by lag-1 behaviour.")
        sel = table("sarimax_selection")
        st.markdown("**SARIMAX settings chosen per city on validation**")
        st.dataframe(sel[sel["chosen"]][["City", "order", "K", "val_MAE"]].round(2), hide_index=True,
                     width="stretch")

    with ph_ml:
        ms_ = table("ml_selection")
        chart(charts.feature_set_chart(ms_, dark),
              ms_.groupby(["model", "feature_set"])["val_rel_MAE"].min().unstack().reset_index(),
              "With AQI history alone, no model beats persistence. Adding **yesterday's pollutants** is "
              "what produces the gain.")
        imp = table("ml_feature_importance")
        which = st.radio("Model", ["xgboost", "random_forest"], horizontal=True, format_func=charts.label,
                         key="imp_model")
        fi = imp[[f"{which}_feature", which]].rename(columns={f"{which}_feature": "feature", which: "importance"})
        fig = charts.importance_chart(fi, "importance", "feature", dark=dark)
        fig.update_layout(title=dict(text=f"{charts.label(which)} feature importance (top 10)", x=0))
        chart(fig, fi)
        pc = table("ml_pooled_vs_per_city")
        st.markdown("**One model for all cities vs one per city** (validation relative MAE, lower = better)")
        st.dataframe(pc[["model", "pooled_val_rel_MAE", "per_city_val_rel_MAE"]]
                     .assign(model=pc["model"].map(charts.label)).round(3)
                     .rename(columns={"pooled_val_rel_MAE": "one model, all cities",
                                      "per_city_val_rel_MAE": "one model per city"}), hide_index=True,
                     width="stretch")
        st.caption("Pooling wins for the tree models; per-city Ridge is marginally better, but pooled is kept "
                   "for all three so they share one setup.")

    with ph_deep:
        ds = table("deep_selection")
        chart(charts.deep_selection_chart(ds, dark), ds,
              "All eight configurations land within about 0.02 of each other; the 14-day window edges out "
              "30 days for both cell types. Epochs were chosen by early stopping on validation.")

    with ph_cmp:
        scores = table("comparison_scores")
        order = list(table("comparison_selection")["model"])
        sp = st.radio("Period", ["test", "test_pre_lockdown", "test_lockdown", "val"], horizontal=True,
                      key="skill_split", format_func=lambda p: {"val": "Validation", "test": "Test (all)",
                                                               "test_pre_lockdown": "Test before lockdown",
                                                               "test_lockdown": "Test during lockdown"}[p])
        chart(charts.skill_heatmap(scores, sp, order, dark),
              (100 * scores[scores["split"] == sp].pivot(index="model", columns="City", values="skill"))
              .loc[order].reset_index())
        top = st.multiselect("Models in the bar charts below", order,
                             default=["persistence", "sarimax_fourier", "ridge", "random_forest", "xgboost",
                                      "lstm", "gru"], format_func=charts.label, key="cmp_models")
        top = [m for m in order if m in top] or ["persistence"]
        all_ = scores[scores["City"] == "ALL"].set_index(["split", "model"])["MAE"]
        pre = 1 - all_[("test_pre_lockdown", "gru")] / all_[("test_pre_lockdown", "persistence")]
        dur = 1 - all_[("test_lockdown", "gru")] / all_[("test_lockdown", "persistence")]
        chart(charts.lockdown_chart(scores, top, dark), note=(
              f"Cleaner air left little to predict: the GRU beats persistence by **{100 * pre:.0f}%** before "
              f"the lockdown but only **{100 * dur:.0f}%** during it."))
        sev = table("comparison_scores_severe")
        chart(charts.severe_chart(sev, top, dark),
              sev[sev["City"] == "ALL"].pivot(index="model", columns="split", values="MAE").reset_index(),
              "On the worst days the tree and recurrent models roughly halve persistence's error.")
        c1, c2 = st.columns(2)
        with c1:
            cm = table("comparison_confusion_final")
            body = cm.set_index(cm.columns[0])
            right = int(np.trace(body.to_numpy()))
            st.markdown(f"**CPCB category: true vs predicted (final model, test)**")
            chart(charts.confusion_heatmap(cm, dark), note=f"Right on **{right}** of **{int(body.to_numpy().sum())}** "
                  "days; almost every miss is a neighbouring category.")
        with c2:
            sh = table("comparison_shap_xgboost")
            st.markdown("**What drives XGBoost (SHAP, test)**")
            chart(charts.shap_group_chart(sh, dark), note="Pollutant lags carry more signal than AQI history.")
            fig = charts.importance_chart(sh, "mean_abs_shap", "feature", top=10, dark=dark)
            chart(fig, sh[["feature", "group", "mean_abs_shap", "share"]])

with tab_board:
    sel = pd.read_csv(TABLES / "comparison_selection.csv")
    st.markdown("#### Ranking (chosen on validation)")
    st.caption("Relative MAE = mean over cities of MAE / persistence MAE. Below 1 beats persistence.")
    st.dataframe(sel.rename(columns={"val_rel_MAE": "val rel. MAE", "test_rel_MAE": "test rel. MAE",
                                     "val_mean_city_MAE": "val MAE", "test_mean_city_MAE": "test MAE"})
                 .round(3), hide_index=True, width="stretch")

    scores = pd.read_csv(TABLES / "comparison_scores.csv")
    period = st.radio("Test period", ["test", "test_pre_lockdown", "test_lockdown"], horizontal=True,
                      format_func=lambda p: {"test": "All of H1 2020", "test_pre_lockdown": "Before lockdown",
                                             "test_lockdown": "During lockdown"}[p])
    tab = scores[scores["split"] == period].pivot(index="model", columns="City", values="MAE").loc[sel["model"]]
    st.markdown(f"#### MAE by city ({period.replace('_', ' ')})")
    st.dataframe(tab.round(1), width="stretch")

    dm = pd.read_csv(TABLES / "comparison_dm.csv")
    dm = dm[dm["ref"] == "persistence"].pivot(index="model", columns="City", values="p_value")
    st.markdown("#### Is the gain over persistence real? (Diebold-Mariano p-values, test)")
    st.caption("Below 0.05 = the difference from persistence is unlikely to be chance.")
    dm = dm.loc[[m for m in sel["model"] if m != "persistence"]]
    st.dataframe(dm.map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"), width="stretch")
    st.caption("Per-city charts of these results are under Model analysis → Comparison.")

with tab_about:
    st.markdown(f"""
**Task.** Forecast tomorrow's AQI for Delhi, Bengaluru, Chennai, Hyderabad and Lucknow from data
up to today. Data: CPCB daily readings 2015-2020 (Kaggle *Air Quality Data in India*).

**Protocol.** Train 2015-2018, validation 2019, test 1 Jan - 1 Jul 2020. Every setting (model
order, features, hyperparameters, epochs) was chosen on validation only. Test forecasts come from
models refitted on train + validation, and test data was never used for a choice.

**Models.** Baselines (persistence, 7-day mean, climatology), SARIMAX and Prophet, Ridge,
Random Forest, XGBoost, LSTM and GRU. The final model is the **LSTM**
(hidden size {lstm['cfg']['hidden']}, {lstm['cfg']['window']}-day window, {lstm['epochs']} epochs).

**Caveats.**
- The test period includes India's COVID-19 lockdown (from {pd.Timestamp(LOCKDOWN_START):%d %b %Y}); pollution
  dropped sharply and the gain over persistence shrank from about 30% to about 8%.
- Only one validation year, so model rankings close together are not reliable.
- The data ends on {clean['Date'].max():%d %b %Y}, so the forecast is for the day after, not today.
""")
    st.caption("Model families: " + ", ".join(f"{m} = {f}" for m, f in FAMILY.items()))
