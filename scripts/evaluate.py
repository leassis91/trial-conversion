# This file is in case we need just a file for evaluating the latest model in the future.

import logging
from datetime import datetime

import pandas as pd

from config.config import MODELS_DIR, RAW_DIR
from src.trial_conversion_model.features import build_features, split_data
from src.trial_conversion_model.predict import evaluate, load_latest_model, save_metrics

logger = logging.getLogger(__name__)


def main() -> None:
    df = pd.read_csv(RAW_DIR / "trials_raw.csv")
    X, y = build_features(df)
    _, X_test, _, y_test = split_data(X, y)
    run_id = datetime.now(datetime.timezone()).strftime("%Y-%m-%d-T%H%M%SZ")
    model = load_latest_model(MODELS_DIR)
    metrics = evaluate(model, X_test=X_test, y_test=y_test)
    save_metrics(metrics, MODELS_DIR, run_id, n_test=len(y_test))


if __name__ == "__main__":
    main()
