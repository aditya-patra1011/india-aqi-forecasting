# India AQI Forecasting

Forecasting next-day Air Quality Index (AQI) for Indian cities and comparing classical time series, machine learning and deep learning models.

## Problem

Supervised time series regression: predict next-day AQI from past pollutant readings and calendar features.

## Dataset

Kaggle: _Air Quality Data in India (2015-2020)_, compiled from CPCB.
Download `city_day.csv` and place it in `data/raw`. The raw data is not committed to this repo.

## Cities

Delhi, Bengaluru, Chennai, Hyderabad, Lucknow

## Split (chronological)

- Train: 2015-01-01 to 2018-12-31
- Validation: 2019
- Test: 2020-01-01 to 2020-07-01

## Project Structure

data/ notebooks/ src/ models/ reports/ app/ tests/

## How to run

From the repo root:

```bash
python -m src.make_dataset    # clean data -> data/processed/aqi_clean.csv
python -m src.eda             # train-only EDA tables and figures
python -m src.run_baselines   # baselines
python -m src.run_classical   # SARIMAX and Prophet
python -m src.run_ml          # Ridge, Random Forest, XGBoost
python -m src.run_deep        # LSTM and GRU (PyTorch)
python -m src.run_comparison  # all models on the same rows, significance, SHAP
python -m src.export_models   # save the final LSTM and XGBoost to models/ for the app
python -m pytest
```

## App

```bash
streamlit run app/streamlit_app.py
```

- **Next-day forecast:** LSTM (final model), XGBoost and persistence for the day after the data ends, with the CPCB category, plus a what-if on the last day's PM2.5 and AQI.
- **History:** actual vs forecasts over validation and test, with the lockdown marked and errors for the selected range.
- **Leaderboard:** the phase 7 ranking, MAE by city before and during the lockdown, and Diebold-Mariano p-values.
- **About:** the method and its caveats.

The app needs the processed data and the exported models, which are not committed. If they are missing, it lists the commands to run.

## Status

- [x] Phase 0: repo setup
- [x] Phase 1: data cleaning
- [x] Phase 2: EDA
- [x] Phase 3: baselines
- [x] Phase 4: classical models
- [x] Phase 5: ML models
- [x] Phase 6: LSTM
- [x] Phase 7: comparison
- [x] Phase 8: app
- [ ] Phase 9: docs
