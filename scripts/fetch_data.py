import logging

from config.config import RAW_DIR
from trial_conversion_model.data import load_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

if __name__ == "__main__":
    load_data(RAW_DIR)
