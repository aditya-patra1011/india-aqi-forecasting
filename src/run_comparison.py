"""Compare every model from phases 3-6 on the same rows. Run from the repo root, after
the earlier run_* scripts have written their prediction files:
    python -m src.run_comparison
The final model is chosen on validation only; test results are reported once.
"""
import ast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src import config
from src.comparison import (FAMILY, LOCKDOWN_START, MODELS, PRED_COLS, add_periods, bucket_confusion,
                            dm_table, merge_predictions, relative_mae_table, score_table, xgboost_shap)
from src.eda import core_pollutants, missing_pct
from src.run_baselines import SEVERE_AQI
from src.split import get_split

TABLES_DIR = config.ROOT / "reports" / "tables"

# reference palette (validated: categorical slots 1-2, blue <-> red diverging, gray midpoint)
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE, RED, MID = "#2a78d6", "#eb6834", "#e34948", "#f0efec"


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(config.DATA_PROCESSED / "aqi_clean.csv", parse_dates=["Date"])
    return df.sort_values(["City", "Date"]).reset_index(drop=True)


def load_prediction_files() -> dict:
    names = sorted({f for f, _ in MODELS.values() if f is not None})
    missing = [n for n in names if not (config.DATA_PROCESSED / n).exists()]
    if missing:
        raise FileNotFoundError(f"Run the earlier phases first; missing {missing}")
    return {n: pd.read_csv(config.DATA_PROCESSED / n, parse_dates=["Date"]) for n in names}


def chosen_xgboost() -> dict:
    sel = pd.read_csv(TABLES_DIR / "ml_selection.csv")
    row = sel[(sel["model"] == "xgboost") & sel["chosen"]].iloc[0]
    return {"feature_set": row["feature_set"], "params": ast.literal_eval(row["params"])}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_skill_heatmap(scores: pd.DataFrame, order: list, split: str = "test", lim: float = 40) -> None:
    """Color is capped at +/- lim % so one very bad baseline doesn't wash out the rest;
    every cell is labelled with its exact value."""
    s = scores[scores["split"] == split]
    tab = 100 * s.pivot(index="model", columns="City", values="skill").loc[order]
    tab = tab[[c for c in tab.columns if c != "ALL"] + ["ALL"]]
    cmap = LinearSegmentedColormap.from_list("div", [RED, MID, BLUE])
    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    ax.imshow(tab.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.iat[i, j]
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.6 * lim else INK)
    ax.set_xticks(range(tab.shape[1]), tab.columns)
    ax.set_yticks(range(tab.shape[0]), [f"{m} ({FAMILY[m]})" for m in tab.index])
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Skill vs persistence on {split}, % lower MAE (blue = better, red = worse)\n"
                 f"rows ordered by validation rank; color capped at ±{lim:.0f}%",
                 color=INK, fontsize=11, loc="left")
    fig.savefig(config.FIGURES_DIR / f"comparison_skill_{split}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_city_test(df: pd.DataFrame, city: str, models: list) -> None:
    g = df[(df["City"] == city) & (df["split"] == "test")]
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.plot(g["Date"], g[config.TARGET], color=INK, linewidth=2, label="actual")
    ax.plot(g["Date"], g[PRED_COLS["persistence"]], color=INK_2, linewidth=1.2, linestyle="--",
            label="persistence")
    for m, color in zip(models, (BLUE, ORANGE)):
        ax.plot(g["Date"], g[PRED_COLS[m]], color=color, linewidth=2, label=m)
    ax.axvline(pd.Timestamp(LOCKDOWN_START), color=INK_2, linewidth=1)
    ax.text(pd.Timestamp(LOCKDOWN_START), ax.get_ylim()[1], " lockdown starts", color=INK_2,
            fontsize=9, va="top")
    ax.set_ylabel("AQI", color=INK_2)
    ax.set_title(f"{city}: next-day forecasts on the test period", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.savefig(config.FIGURES_DIR / f"comparison_{city.lower()}_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap(shap_tab: pd.DataFrame, top: int = 12) -> None:
    t = shap_tab.head(top).iloc[::-1]
    groups = shap_tab.groupby("group")["share"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.barh(t["feature"], t["mean_abs_shap"], color=BLUE, height=0.6)
    for y, v in enumerate(t["mean_abs_shap"]):
        ax.text(v, y, f" {v:.1f}", va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("mean |SHAP| (AQI points)", color=INK_2)
    shares = ", ".join(f"{g} {100 * s:.0f}%" for g, s in groups.items())
    ax.set_title(f"XGBoost on test: what drives the forecast\nshare by group: {shares}",
                 color=INK, fontsize=11, loc="left")
    fig.savefig(config.FIGURES_DIR / "comparison_shap_xgboost.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = add_periods(merge_predictions(load_clean(), load_prediction_files()))
    scores = score_table(df)
    severe = score_table(df, min_true_aqi=SEVERE_AQI)
    selection = relative_mae_table(scores, "val").merge(
        relative_mae_table(scores, "test")[["model", "test_rel_MAE", "test_mean_city_MAE"]], on="model")
    final = selection.iloc[0]["model"]
    best = selection.groupby("family", sort=False).head(1).set_index("family")["model"]

    models = [m for m in MODELS if m != "persistence"]
    dm_persist = dm_table(df, models, ref="persistence", split="test")
    dm_pair = dm_table(df, [best["deep"]], ref=best["ml"], split="test")
    confusion = bucket_confusion(df, PRED_COLS[final], "test")
    core = core_pollutants(missing_pct(get_split(df, "train")))
    shap_tab = xgboost_shap(df, core, chosen_xgboost(), "test")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(TABLES_DIR / "comparison_scores.csv", index=False)
    severe.to_csv(TABLES_DIR / "comparison_scores_severe.csv", index=False)
    selection.to_csv(TABLES_DIR / "comparison_selection.csv", index=False)
    pd.concat([dm_persist, dm_pair]).to_csv(TABLES_DIR / "comparison_dm.csv", index=False)
    confusion.to_csv(TABLES_DIR / "comparison_confusion_final.csv")
    shap_tab.to_csv(TABLES_DIR / "comparison_shap_xgboost.csv", index=False)

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_skill_heatmap(scores, list(selection["model"]))
    plot_city_test(df, "Delhi", [best["ml"], best["deep"]])
    plot_shap(shap_tab)

    pd.set_option("display.width", 220)
    print("=== Ranking on validation (mean over cities of MAE / persistence MAE) ===")
    print(selection.round(3).to_string(index=False))
    print(f"\nFinal model (validation only): {final}. Best per family: {best.to_dict()}")
    for period in ("val", "test", "test_pre_lockdown", "test_lockdown"):
        tab = scores[scores["split"] == period].pivot(index="model", columns="City", values="MAE")
        print(f"\n=== MAE by model and city ({period}, n per city = "
              f"{scores[(scores['split'] == period) & (scores['model'] == 'persistence')].set_index('City')['n'].to_dict()}) ===")
        print(tab.loc[selection["model"]].round(1).to_string())
    print(f"\n=== MAE on severe days (actual AQI >= {SEVERE_AQI}), ALL cities ===")
    sv = severe[severe["City"] == "ALL"].pivot(index="model", columns="split", values="MAE")
    print(sv.loc[selection["model"], ["val", "test"]].round(1).to_string())
    print("\n=== Diebold-Mariano vs persistence on test (mean |e| difference; negative = better) ===")
    ranked = [m for m in selection["model"] if m != "persistence"]
    for value in ("mean_loss_diff", "p_value"):
        print(f"--- {value} ---")
        print(dm_persist.pivot(index="model", columns="City", values=value).loc[ranked].round(3).to_string())
    print(f"\n=== Diebold-Mariano {best['deep']} vs {best['ml']} on test ===")
    print(dm_pair.round(3).to_string(index=False))
    print(f"\n=== CPCB category confusion, {final}, test ===")
    print(confusion.to_string())
    print("\n=== XGBoost SHAP on test: share by feature group ===")
    print(shap_tab.groupby("group")["share"].sum().sort_values(ascending=False).round(3).to_string())
    print(shap_tab.head(10).round(3).to_string(index=False))
    print(f"\nTables saved to {TABLES_DIR}\nFigures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
