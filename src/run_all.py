"""Run the whole pipeline, from raw data to the app's models. Run from the repo root:
    python -m src.run_all
Stages run in order, since each one reads what the previous ones wrote. Takes about
15 minutes on a laptop CPU, most of it the LSTM/GRU grid.
    python -m src.run_all --from run_ml     # restart from one stage
"""
import argparse
import importlib
import time

from src import config

STAGES = [
    ("make_dataset", "clean data -> data/processed/aqi_clean.csv"),
    ("eda", "train-only EDA tables and figures"),
    ("run_baselines", "baselines"),
    ("run_classical", "SARIMAX and Prophet"),
    ("run_ml", "Ridge, Random Forest, XGBoost"),
    ("run_deep", "LSTM and GRU"),
    ("run_comparison", "all models on the same rows, significance, SHAP"),
    ("export_models", "final models for the app -> models/"),
]


def main() -> None:
    names = [s for s, _ in STAGES]
    parser = argparse.ArgumentParser(description="Run the pipeline stages in order.")
    parser.add_argument("--from", dest="start", choices=names, default=names[0],
                        help="first stage to run (earlier stages' outputs must exist)")
    args = parser.parse_args()
    if not config.DATA_RAW.exists():
        raise SystemExit(f"Raw data not found at {config.DATA_RAW}. Download city_day.csv first (see README).")

    total = time.time()
    for name, what in STAGES[names.index(args.start):]:
        print(f"\n{'=' * 70}\n>>> {name}: {what}\n{'=' * 70}", flush=True)
        start = time.time()
        importlib.import_module(f"src.{name}").main()
        print(f">>> {name} done in {time.time() - start:.0f}s", flush=True)
    print(f"\nPipeline finished in {(time.time() - total) / 60:.1f} min")


if __name__ == "__main__":
    main()
