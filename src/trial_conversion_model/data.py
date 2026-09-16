import logging
import os
# from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine

# GLOBAL VARIABLES
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
CURRENT_TABLE = os.getenv("CURRENT_TABLE")


_FULL_QUERY = f"""
SELECT *
FROM {CURRENT_TABLE}"""

logger = logging.getLogger(__name__)


def load_data():
    engine = create_engine(
        f"postgresql://students:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    logger.info("Fetching data...")
    df = pd.read_sql(_FULL_QUERY, engine)
    logger.info("Saving...")
    df.to_csv("../data/01_raw/trials_raw.csv", index=False)
    logger.info("Data saved successfully!")


# def read_data(filename) -> pd.DataFrame:
#     input_path = Path(__file__).parent / f"data/01_raw/{filename}.pkl"
#     df = pd.read_csv(input_path)

#     return df
