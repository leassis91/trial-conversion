import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

# GLOBAL VARIABLES
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
CURRENT_TABLE = os.getenv("CURRENT_TABLE")


_FULL_QUERY = f"""
SELECT *
FROM {CURRENT_TABLE}"""

logger = logging.getLogger(__name__)


def load_data(output_dir: Path) -> None:
    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    logger.info("Fetching data...")
    df = pd.read_sql(_FULL_QUERY, engine)
    logger.info("Saving...")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{output_dir}/trials_raw.csv", index=False)
    logger.info("Data saved successfully!")
