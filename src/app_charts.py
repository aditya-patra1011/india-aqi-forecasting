"""Interactive charts for the app's analysis tabs, built from the committed tables in
reports/tables (plus the clean data for the AQI distribution). Each function returns a
plotly Figure; colors follow one fixed mapping per city / model / feature set, so an
entity keeps its color on every chart.
"""
import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import config
from src.cleaning import BUCKET_LABELS, aqi_to_bucket

TABLES = config.ROOT / "reports" / "tables"

# validated categorical palette (light / dark steps of the same hues)
SLOTS = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
         "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]}
MODEL_ORDER = ["xgboost", "lstm", "random_forest", "gru", "sarimax_fourier"]
GRAY = "#8a8984"
LABELS = {"persistence": "Persistence", "lstm": "LSTM", "gru": "GRU", "xgboost": "XGBoost",
          "random_forest": "Random Forest", "sarimax_fourier": "SARIMAX", "prophet_lag1": "Prophet",
          "ridge": "Ridge", "mean_7d": "7-day mean", "monthly_climatology": "Climatology",
          "seasonal_naive_7d": "Seasonal naive (7d)", "seasonal_naive_365d": "Seasonal naive (365d)"}
FEATURE_SETS = {"A_aqi_only": "AQI history only", "B_core": "+ core pollutants", "C_all": "+ all pollutants"}
CATEGORY_COLORS = {"Good": "#00b050", "Satisfactory": "#92d050", "Moderate": "#ffd500",
                   "Poor": "#ff9900", "Very Poor": "#e02020", "Severe": "#8b0000"}


def _slots(dark: bool) -> list:
    return SLOTS["dark" if dark else "light"]


def city_colors(dark: bool = False) -> dict:
    return dict(zip(config.CITIES, _slots(dark)))


def model_colors(dark: bool = False) -> dict:
    out = dict(zip(MODEL_ORDER, _slots(dark)))
    out["persistence"] = GRAY
    return out


def label(model: str) -> str:
    return LABELS.get(model, model)


def _diverging(dark: bool) -> list:
    mid = "#383835" if dark else "#f0efec"
    return [[0, "#e34948"], [0.5, mid], [1, "#3987e5" if dark else "#2a78d6"]]


def _sequential(dark: bool) -> list:
    return [[0, "#184f95"], [1, "#cde2fb"]] if dark else [[0, "#cde2fb"], [1, "#0d366b"]]


def _layout(fig: go.Figure, title: str = "", y_title: str = "", height: int = 380, **kw) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15)), height=height,
                      margin=dict(l=10, r=10, t=60 if title else 20, b=10), yaxis_title=y_title,
                      legend=dict(orientation="h", y=-0.18, x=0), hoverlabel=dict(namelength=-1), **kw)
    return fig


def show_only(fig: go.Figure, names) -> go.Figure:
    """Hide every trace not in `names` but keep it in the legend, so a click brings it back."""
    keep = set(names)
    for tr in fig.data:
        tr.visible = True if tr.name in keep else "legendonly"
    return fig


def read(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv", **kw)


# ---------- data & EDA ----------

def profile_chart(table: pd.DataFrame, by: str, dark: bool = False) -> go.Figure:
    """Mean train-split AQI per city by month, weekday or year (one line per city)."""
    key = table.columns[0]
    names = {"month": lambda v: calendar.month_abbr[int(v)], "weekday": lambda v: calendar.day_abbr[int(v)],
             "year": lambda v: str(int(v))}[by]
    x = [names(v) for v in table[key]]
    fig = go.Figure()
    for city, color in city_colors(dark).items():
        fig.add_scatter(x=x, y=table[city], name=city, mode="lines+markers",
                        line=dict(color=color, width=2), marker=dict(size=8),
                        hovertemplate=f"{city}: %{{y:.0f}}<extra></extra>")
    return _layout(fig, f"Mean AQI by {by} (train 2015-2018)", "AQI", hovermode="x unified")


def category_share_chart(clean: pd.DataFrame) -> go.Figure:
    """Share of train days in each CPCB category, per city (stacked to 100%)."""
    train = clean[(clean["split"] == "train") & clean[config.TARGET].notna()]
    cats = pd.Categorical(aqi_to_bucket(train[config.TARGET]), categories=BUCKET_LABELS)
    share = pd.crosstab(train["City"], cats, normalize="index").reindex(config.CITIES) * 100
    fig = go.Figure()
    for cat in BUCKET_LABELS:
        fig.add_bar(y=share.index, x=share[cat], name=cat, orientation="h",
                    marker=dict(color=CATEGORY_COLORS[cat], line=dict(width=1, color="rgba(0,0,0,0)")),
                    hovertemplate=f"%{{y}} · {cat}: %{{x:.1f}}% of days<extra></extra>")
    return _layout(fig, "Share of days in each CPCB category (train)", barmode="stack",
                   xaxis=dict(title="% of days", range=[0, 100]))


def missing_heatmap(missing: pd.DataFrame, dark: bool = False) -> go.Figure:
    """% of train rows missing, per column and city."""
    t = missing.set_index(missing.columns[0])
    fig = go.Figure(go.Heatmap(z=t.to_numpy(), x=list(t.columns), y=list(t.index), zmin=0, zmax=100,
                               colorscale=_sequential(dark), texttemplate="%{z:.0f}",
                               hovertemplate="%{y} in %{x}: %{z:.1f}% missing<extra></extra>",
                               colorbar=dict(title="% missing")))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Missing values on train, % of days", height=460)


def autocorr_chart(autocorr: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Autocorrelation of AQI at each lag, one line per city."""
    t = autocorr.set_index(autocorr.columns[0])
    x = [f"{int(v)}d" for v in t.index]
    fig = go.Figure()
    for city, color in city_colors(dark).items():
        fig.add_scatter(x=x, y=t[city], name=city, mode="lines+markers", line=dict(color=color, width=2),
                        marker=dict(size=8), hovertemplate=f"{city}: %{{y:.2f}}<extra></extra>")
    return _layout(fig, "How much today's AQI resembles AQI n days earlier", "autocorrelation",
                   hovermode="x unified", xaxis_title="lag", yaxis=dict(range=[0, 1]))


def lagged_corr_heatmap(corr: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Correlation of today's AQI with each pollutant yesterday, per city."""
    t = corr.set_index(corr.columns[0])
    fig = go.Figure(go.Heatmap(z=t.to_numpy(), x=list(t.columns), y=list(t.index), zmin=-1, zmax=1,
                               colorscale=_diverging(dark), texttemplate="%{z:.2f}",
                               hovertemplate="%{y} yesterday vs AQI today, %{x}: %{z:.2f}<extra></extra>",
                               colorbar=dict(title="r")))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Correlation of today's AQI with yesterday's pollutant (train)", height=460)


# ---------- models ----------

def city_mae_chart(results: pd.DataFrame, split: str, models: list, dark: bool = False,
                   title: str = "") -> go.Figure:
    """Grouped bars: MAE per city for each model on one split."""
    r = results[results["split"] == split]
    colors = model_colors(dark)
    # models without a fixed color (other baselines, Prophet) take the remaining palette hues first,
    # then any hue not already used in this chart (8 hues for 10 models, so some reuse is unavoidable)
    used = {colors[m] for m in models if m in colors}
    spare = ["#008300", "#6250d6", "#e34948", *_slots(dark)]
    extra = iter([c for c in dict.fromkeys(spare) if c not in used])
    fig = go.Figure()
    for m in models:
        g = r[r["model"] == m].set_index("City").reindex([*config.CITIES, "ALL"])
        color = colors.get(m) or next(extra)
        fig.add_bar(x=g.index, y=g["MAE"], name=label(m), marker_color=color,
                    hovertemplate=f"{label(m)}, %{{x}}: MAE %{{y:.1f}}<extra></extra>")
    return _layout(fig, title, "MAE (AQI points)", barmode="group", bargap=0.25, bargroupgap=0.08)


def feature_set_chart(selection: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Best validation relative MAE per ML model for each feature set."""
    best = selection.groupby(["model", "feature_set"])["val_rel_MAE"].min().unstack()
    best = best.reindex(["ridge", "random_forest", "xgboost"])
    fig = go.Figure()
    for fs, color in zip(FEATURE_SETS, _slots(dark)):
        fig.add_bar(x=[label(m) for m in best.index], y=best[fs], name=FEATURE_SETS[fs], marker_color=color,
                    hovertemplate=f"%{{x}}, {FEATURE_SETS[fs]}: %{{y:.3f}}<extra></extra>")
    fig.add_hline(y=1, line=dict(color=GRAY, dash="dash"), annotation_text="persistence",
                  annotation_position="top left")
    return _layout(fig, "Validation error relative to persistence, by feature set (below 1 = better)",
                   "relative MAE", barmode="group", yaxis=dict(range=[0.7, 1.1]))


def importance_chart(table: pd.DataFrame, value: str, name: str, top: int = 12,
                     dark: bool = False) -> go.Figure:
    """Horizontal bars of one importance column (largest on top)."""
    t = table.nlargest(top, value).iloc[::-1]
    fig = go.Figure(go.Bar(x=t[value], y=t[name], orientation="h", marker_color=_slots(dark)[0],
                           text=t[value].map(lambda v: f"{v:.3g}"), textposition="outside",
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    return _layout(fig, height=max(300, 28 * len(t) + 80))


def deep_selection_chart(selection: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Validation relative MAE of every LSTM / GRU configuration tried."""
    colors = model_colors(dark)
    fig = go.Figure()
    for cell in ("lstm", "gru"):
        s = selection[selection["cell"] == cell]
        x = [f"{h} units · {w}-day" for h, w in zip(s["hidden"], s["window"])]
        fig.add_bar(x=x, y=s["val_rel_MAE"], name=label(cell), marker_color=colors[cell],
                    marker_line=dict(width=[3 if c else 0 for c in s["chosen"]], color=GRAY),
                    customdata=np.c_[s["epochs"], s["chosen"]],
                    hovertemplate=(f"{label(cell)} %{{x}}: %{{y:.3f}}<br>epochs %{{customdata[0]}}"
                                   "<br>chosen %{customdata[1]}<extra></extra>"))
    return _layout(fig, "Validation relative MAE per configuration (outlined = chosen)", "relative MAE",
                   barmode="group", yaxis=dict(range=[0.75, 0.82]))


def skill_heatmap(scores: pd.DataFrame, split: str, order: list, dark: bool = False, cap: float = 40) -> go.Figure:
    """% lower MAE than persistence per model and city; color capped at +/- cap."""
    s = scores[scores["split"] == split]
    t = (100 * s.pivot(index="model", columns="City", values="skill")).loc[order]
    t = t[[*config.CITIES, "ALL"]]
    fig = go.Figure(go.Heatmap(z=t.clip(-cap, cap).to_numpy(), x=list(t.columns), y=[label(m) for m in t.index],
                               customdata=t.to_numpy(), zmin=-cap, zmax=cap, colorscale=_diverging(dark),
                               text=t.map(lambda v: f"{v:+.0f}").to_numpy(), texttemplate="%{text}",
                               hovertemplate="%{y}, %{x}: %{customdata:+.1f}% vs persistence<extra></extra>",
                               colorbar=dict(title="skill %")))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, f"Skill vs persistence, % lower MAE (color capped at ±{cap:.0f}%)", height=480)


def lockdown_chart(scores: pd.DataFrame, models: list, dark: bool = False) -> go.Figure:
    """Pooled MAE before vs during the lockdown, per model."""
    s = scores[scores["City"] == "ALL"].pivot(index="model", columns="split", values="MAE").loc[models]
    fig = go.Figure()
    for period, name, color in (("test_pre_lockdown", "Before lockdown (1 Jan - 24 Mar)", _slots(dark)[0]),
                                ("test_lockdown", "During lockdown (25 Mar - 1 Jul)", _slots(dark)[1])):
        fig.add_bar(x=[label(m) for m in s.index], y=s[period], name=name, marker_color=color,
                    hovertemplate=f"%{{x}}, {name}: MAE %{{y:.1f}}<extra></extra>")
    return _layout(fig, "Test MAE before vs during the COVID-19 lockdown (all cities)", "MAE (AQI points)",
                   barmode="group")


def severe_chart(severe: pd.DataFrame, models: list, dark: bool = False) -> go.Figure:
    """MAE on days with AQI >= 300, validation vs test, pooled."""
    s = severe[severe["City"] == "ALL"].pivot(index="model", columns="split", values="MAE").loc[models]
    n = severe[(severe["City"] == "ALL") & (severe["model"] == "persistence")].set_index("split")["n"]
    fig = go.Figure()
    for split, color in (("val", _slots(dark)[0]), ("test", _slots(dark)[1])):
        name = f"{'Validation' if split == 'val' else 'Test'} (n = {int(n.get(split, 0))} days)"
        fig.add_bar(x=[label(m) for m in s.index], y=s[split], name=name, marker_color=color,
                    hovertemplate=f"%{{x}}, {name}: MAE %{{y:.1f}}<extra></extra>")
    return _layout(fig, "Error on severe days (actual AQI ≥ 300)", "MAE (AQI points)", barmode="group")


def confusion_heatmap(confusion: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Counts of true vs predicted CPCB category."""
    t = confusion.set_index(confusion.columns[0])[BUCKET_LABELS].reindex(BUCKET_LABELS)
    fig = go.Figure(go.Heatmap(z=t.to_numpy(), x=BUCKET_LABELS, y=BUCKET_LABELS, colorscale=_sequential(dark),
                               texttemplate="%{z}", hovertemplate="true %{y}, predicted %{x}: %{z} days<extra></extra>",
                               colorbar=dict(title="days")))
    fig.update_yaxes(autorange="reversed", title="true category")
    fig.update_xaxes(title="predicted category")
    return _layout(fig, height=440)


def shap_group_chart(shap_table: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Share of XGBoost's SHAP importance per feature group."""
    g = shap_table.groupby("group")["share"].sum().sort_values() * 100
    fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=_slots(dark)[0],
                           text=[f"{v:.0f}%" for v in g.values], textposition="outside",
                           hovertemplate="%{y}: %{x:.1f}% of total |SHAP|<extra></extra>"))
    return _layout(fig, "Where XGBoost's signal comes from", height=260, xaxis=dict(range=[0, 75]))
