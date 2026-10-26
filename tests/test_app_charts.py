import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src import app_charts as charts
from src import config


def _names(fig):
    return [t.name for t in fig.data]


@pytest.mark.parametrize("name,by", [("monthly_profile", "month"), ("weekday_profile", "weekday"),
                                     ("yearly_profile", "year")])
def test_profile_chart_has_one_line_per_city(name, by):
    fig = charts.profile_chart(charts.read(name), by)
    assert _names(fig) == config.CITIES
    assert len(fig.data[0].x) == len(charts.read(name))


def test_city_and_model_colors_are_fixed_and_distinct():
    for dark in (False, True):
        c = charts.city_colors(dark)
        assert list(c) == config.CITIES and len(set(c.values())) == len(c)
        m = charts.model_colors(dark)
        assert m["persistence"] == charts.GRAY and len(set(m.values())) == len(m)


def test_city_mae_chart_gives_every_model_its_own_color_even_with_all_baselines():
    models = ["persistence", "mean_7d", "seasonal_naive_7d", "seasonal_naive_365d", "monthly_climatology"]
    fig = charts.city_mae_chart(charts.read("baseline_results"), "val", models)
    colors = [t.marker.color for t in fig.data]
    assert len(colors) == len(models) and len(set(colors)) == len(models)
    assert colors[0] == charts.GRAY                      # persistence keeps its fixed color


def test_show_only_hides_other_traces_but_keeps_them_in_the_legend():
    fig = charts.show_only(charts.autocorr_chart(charts.read("autocorr")), ["Delhi", "Lucknow"])
    vis = {t.name: t.visible for t in fig.data}
    assert vis["Delhi"] is True and vis["Lucknow"] is True
    assert vis["Chennai"] == "legendonly"


def test_heatmaps_read_the_tables_with_their_index_columns():
    miss = charts.read("missing_pct")
    fig = charts.missing_heatmap(miss)
    assert fig.data[0].z.shape == (len(miss), len(config.CITIES))
    corr = charts.lagged_corr_heatmap(charts.read("lagged_corr_lag1"))
    assert np.nanmax(np.abs(corr.data[0].z)) <= 1


def test_category_share_rows_sum_to_100():
    rng = np.random.default_rng(0)
    clean = pd.DataFrame({"City": np.repeat(config.CITIES, 50), "split": "train",
                          "AQI": rng.uniform(0, 500, 250)})
    fig = charts.category_share_chart(clean)
    total = sum(np.asarray(t.x, float) for t in fig.data)
    assert np.allclose(total, 100)


def test_model_charts_build_from_the_committed_tables():
    order = list(charts.read("comparison_selection")["model"])
    scores = charts.read("comparison_scores")
    figs = [
        charts.city_mae_chart(charts.read("baseline_results"), "val", ["persistence", "mean_7d"]),
        charts.feature_set_chart(charts.read("ml_selection")),
        charts.deep_selection_chart(charts.read("deep_selection")),
        charts.skill_heatmap(scores, "test", order),
        charts.lockdown_chart(scores, ["persistence", "lstm"]),
        charts.severe_chart(charts.read("comparison_scores_severe"), ["persistence", "lstm"]),
        charts.confusion_heatmap(charts.read("comparison_confusion_final")),
        charts.shap_group_chart(charts.read("comparison_shap_xgboost")),
    ]
    assert all(isinstance(f, go.Figure) and len(f.data) > 0 for f in figs)


def test_skill_heatmap_caps_color_but_keeps_exact_values():
    scores = charts.read("comparison_scores")
    order = list(charts.read("comparison_selection")["model"])
    fig = charts.skill_heatmap(scores, "test", order, cap=40)
    z, exact = np.asarray(fig.data[0].z, float), np.asarray(fig.data[0].customdata, float)
    assert z.max() <= 40 and z.min() >= -40
    assert exact.min() < -40                       # climatology's real value survives in the hover


def test_feature_set_chart_shows_aqi_only_near_persistence():
    fig = charts.feature_set_chart(charts.read("ml_selection"))
    aqi_only = next(t for t in fig.data if t.name == charts.FEATURE_SETS["A_aqi_only"])
    assert all(0.95 < v < 1.1 for v in aqi_only.y)
