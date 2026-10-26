"""India AQI forecasting app. Run from the repo root:
    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import app_logic, config
from src.comparison import FAMILY, LOCKDOWN_START, PRED_COLS
from src.evaluation import regression_metrics

TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures"

# fixed color per model (validated categorical order), so a model keeps its color
# whatever else is selected
MODEL_COLORS = {"xgboost": "#3987e5", "lstm": "#eb6834", "random_forest": "#1baf7a",
                "gru": "#eda100", "sarimax_fourier": "#e87ba4", "persistence": "#8a8984"}
LABELS = {"persistence": "Persistence", "lstm": "LSTM", "gru": "GRU", "xgboost": "XGBoost",
          "random_forest": "Random Forest", "sarimax_fourier": "SARIMAX"}

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

tab_fc, tab_hist, tab_board, tab_about = st.tabs(["Next-day forecast", "History", "Leaderboard", "About"])

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
        table = pd.DataFrame({"model": ["LSTM", "XGBoost", "Persistence"],
                              "actual data": [row["lstm"], row["xgboost"], row["persistence"]],
                              "what-if": [arow["lstm"], arow["xgboost"], arow["persistence"]]})
        table["change"] = table["what-if"] - table["actual data"]
        table["category (what-if)"] = table["what-if"].map(app_logic.category)
        st.dataframe(table.round(1), hide_index=True, use_container_width=True)
    else:
        st.info("Move a slider to compare.")

with tab_hist:
    h = history[(history["City"] == city) & history["split"].isin(["val", "test"])]
    lo, hi = h["Date"].min().date(), h["Date"].max().date()
    d1, d2 = st.slider("Date range", lo, hi, (pd.Timestamp("2020-01-01").date(), hi), format="DD MMM YYYY")
    chosen = st.multiselect("Models", list(MODEL_COLORS), default=["persistence", "xgboost", "lstm"],
                            format_func=LABELS.get)
    h = h[(h["Date"].dt.date >= d1) & (h["Date"].dt.date <= d2)]
    dark = st.context.theme.type == "dark"
    ink, ink_2 = ("#f0efec", "#c3c2b7") if dark else ("#0b0b0b", "#52514e")

    fig = go.Figure()
    fig.add_scatter(x=h["Date"], y=h[config.TARGET], name="Actual", line=dict(color=ink, width=2.5))
    for m in chosen:
        dash = "dash" if m == "persistence" else "solid"
        fig.add_scatter(x=h["Date"], y=h[PRED_COLS[m]], name=LABELS[m],
                        line=dict(color=MODEL_COLORS[m], width=2, dash=dash))
    if d1 <= pd.Timestamp(LOCKDOWN_START).date() <= d2:
        fig.add_vline(x=pd.Timestamp(LOCKDOWN_START), line=dict(color=ink_2, width=1))
        fig.add_annotation(x=pd.Timestamp(LOCKDOWN_START), y=1, yref="paper", text="lockdown starts",
                           showarrow=False, xanchor="left", font=dict(color=ink_2, size=11))
    fig.update_layout(hovermode="x unified", height=450, margin=dict(l=10, r=10, t=70, b=10),
                      yaxis_title="AQI", legend=dict(orientation="h", y=1.2, x=0),
                      plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    st.plotly_chart(fig, use_container_width=True)

    if chosen:
        rows = [{"model": LABELS[m], **regression_metrics(h[config.TARGET], h[PRED_COLS[m]])} for m in chosen]
        err = pd.DataFrame(rows).rename(columns={"bucket_acc": "category accuracy", "MAPE": "MAPE %"})
        st.caption(f"Errors over the selected {len(h)} days")
        st.dataframe(err.round(2), hide_index=True, use_container_width=True)

with tab_board:
    sel = pd.read_csv(TABLES / "comparison_selection.csv")
    st.markdown("#### Ranking (chosen on validation)")
    st.caption("Relative MAE = mean over cities of MAE / persistence MAE. Below 1 beats persistence.")
    st.dataframe(sel.rename(columns={"val_rel_MAE": "val rel. MAE", "test_rel_MAE": "test rel. MAE",
                                     "val_mean_city_MAE": "val MAE", "test_mean_city_MAE": "test MAE"})
                 .round(3), hide_index=True, use_container_width=True)

    scores = pd.read_csv(TABLES / "comparison_scores.csv")
    period = st.radio("Test period", ["test", "test_pre_lockdown", "test_lockdown"], horizontal=True,
                      format_func=lambda p: {"test": "All of H1 2020", "test_pre_lockdown": "Before lockdown",
                                             "test_lockdown": "During lockdown"}[p])
    tab = scores[scores["split"] == period].pivot(index="model", columns="City", values="MAE").loc[sel["model"]]
    st.markdown(f"#### MAE by city ({period.replace('_', ' ')})")
    st.dataframe(tab.round(1), use_container_width=True)

    dm = pd.read_csv(TABLES / "comparison_dm.csv")
    dm = dm[dm["ref"] == "persistence"].pivot(index="model", columns="City", values="p_value")
    st.markdown("#### Is the gain over persistence real? (Diebold-Mariano p-values, test)")
    st.caption("Below 0.05 = the difference from persistence is unlikely to be chance.")
    dm = dm.loc[[m for m in sel["model"] if m != "persistence"]]
    st.dataframe(dm.map(lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"), use_container_width=True)
    st.image(str(FIGURES / "comparison_skill_test.png"), caption="Skill vs persistence on test")

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
