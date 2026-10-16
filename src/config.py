"""Central configuration. Every split date and path lives here so that
no script can define its own (and accidentally leak) a different split."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "city_day.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# Cities with complete daily coverage and low AQI missingness
CITIES = ["Delhi", "Bengaluru", "Chennai", "Hyderabad", "Lucknow"]

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]
TARGET = "AQI"
BUCKET_COL = "AQI_Bucket"

# Chronological split (inclusive end dates). Test = everything after VAL_END,
# i.e. 2020-01-01 to 2020-07-01, which overlaps the COVID-19 lockdown.
TRAIN_END = "2018-12-31"
VAL_END = "2019-12-31"

HORIZON_DAYS = 1
RANDOM_STATE = 42