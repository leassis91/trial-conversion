from datetime import datetime

import pandas as pd

from src.trial_conversion_model.features import build_features, split_data
from src.trial_conversion_model.train import save_model, train_xgb

RAW_DATA_DIR = "../data/01_raw"
NOW = datetime.now(datetime.utc)

if __name__ == "__main__":
    df = pd.read_csv(RAW_DATA_DIR / "trials_raw.csv")
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_xgb(X_train, y_train)
    save_model(model, f"xgb-{NOW}")
