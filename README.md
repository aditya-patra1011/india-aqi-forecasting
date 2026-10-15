# India Aqi Forecasting

Forecasting next-day Air Quality Index (AQI) for Indian cities and comparing classical time series, machine learning and deep learning models.

## Problem

Supervised time series regression: predict next-day AQI from past pollutant readings and calendar features

## Dataset

Kaggle: _Air Quality Data in India (2015-2020)_, compiled from CPCB
Download `city_day.csv` and place it in `data/raw`. The raw data is not committed to this repo

## Cities

Delhi, Bengaluru, Chennai, Hyderbad, Lucknow

# Split (chronological)

- Train: 2015-01-01 to 2018-12-31
- Validation: 2019
- Test: 2020-01-01 to 2020-07-01

## Project Structure

data/ notebooks/ src/ models/ reports/ app/ tests/

## Status

- [x]: Phase 0: repo setup
- [ ]: Phase 1: data cleaning
- [ ]: Phase 2: EDA
- [ ]: Phase 3: baselines
- [ ]: Phase 4: classical models
- [ ]: Phase 5: ML models
- [ ]: Phase 6: LSTM
- [ ]: Phase 7: comparison
- [ ]: Phase 8: app
- [ ]: Phase 9: docs
