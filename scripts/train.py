from datetime import datetime

import pandas as pd

from config.config import MODELS_DIR, RAW_DIR
from src.trial_conversion_model.features import build_features, split_data
from src.trial_conversion_model.predict import predict, save_metrics
from src.trial_conversion_model.train import save_model, train_xgb

if __name__ == "__main__":
    df = pd.read_csv(RAW_DIR / "trials_raw.csv")
    X, y = build_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%SZ")
    model = train_xgb(X_train, y_train)
    save_model(model, f"xgb-{run_id}", MODELS_DIR)
    metrics = predict(model, X_test=X_test, y_test=y_test)
    # save_metrics(metrics, MODELS_DIR, run_id)
    save_metrics(metrics, MODELS_DIR, run_id, n_test=len(y_test))
