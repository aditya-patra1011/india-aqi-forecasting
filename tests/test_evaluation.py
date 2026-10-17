import numpy as np
import pandas as pd
import pytest

from src.evaluation import bucket_accuracy, evaluate, regression_metrics


def test_regression_metrics_known_values():
    m = regression_metrics([100, 200], [110, 180])
    assert m["n"] == 2
    assert m["MAE"] == pytest.approx(15.0)
    assert m["RMSE"] == pytest.approx(np.sqrt((100 + 400) / 2))
    assert m["MAPE"] == pytest.approx(10.0)                      # mean(10/100, 20/200) * 100


def test_regression_metrics_drops_nan_pairs_and_reports_n():
    m = regression_metrics([100, np.nan, 300], [100, 50, np.nan])
    assert m["n"] == 1
    assert m["MAE"] == 0.0


def test_regression_metrics_all_nan_returns_nan_not_error():
    m = regression_metrics([np.nan], [np.nan])
    assert m["n"] == 0 and np.isnan(m["MAE"])


def test_bucket_accuracy_clips_predictions_to_valid_range():
    # true 450 (Severe); predicted 900 clips to 500 (Severe) -> correct; predicted 40 (Good) -> wrong
    assert bucket_accuracy([450, 450], [900, 40]) == pytest.approx(0.5)


def _scored_frame():
    return pd.DataFrame({
        "City": ["A", "A", "B", "B", "A"],
        "split": ["val", "val", "val", "val", "test"],
        "AQI": [100.0, 400.0, 200.0, 50.0, 100.0],
        "pred": [110.0, 380.0, 200.0, 70.0, 0.0],
    })


def test_evaluate_scores_per_city_pooled_and_by_split():
    out = evaluate(_scored_frame(), {"m": "pred"}, "val").set_index("City")
    assert out.loc["A", "MAE"] == pytest.approx(15.0)
    assert out.loc["B", "MAE"] == pytest.approx(10.0)
    assert out.loc["ALL", "n"] == 4
    assert out.loc["ALL", "MAE"] == pytest.approx((10 + 20 + 0 + 20) / 4)   # test row excluded


def test_evaluate_severe_slice_keeps_only_high_actual_aqi():
    out = evaluate(_scored_frame(), {"m": "pred"}, "val", min_true_aqi=300).set_index("City")
    assert out.loc["ALL", "n"] == 1
    assert out.loc["ALL", "MAE"] == pytest.approx(20.0)