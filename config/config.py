# configuration and constant variables file

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "01_raw"

MODELS_DIR = ROOT_DIR / "models"

ID_COL = "trial_id"
# FEATURES = [
#     "sessions_3d",
#     "active_days_3d",
#     "day1_share",
#     "listen_share",
#     "avg_session_minutes",
#     "total_minutes_3d",
#     "country",
#     "device_type",
# ]
TARGET = "converted"
SEED = 42
