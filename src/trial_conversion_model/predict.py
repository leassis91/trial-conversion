import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:

    probs = model.predict_proba(X_test)[:, 1]
    logger.info(f"Evaluating model {str(model)[:3]} on {len(y_test)} rows...")
    auc = round(roc_auc_score(y_test, probs), 4)
    logger.info(f"Metrics: {auc}")

    return {"AUC": auc}


def save_metrics(metrics: dict, output_dir: Path, run_id: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "metrics": metrics}
    path = output_dir / f"metrics-{run_id}.json"
    logger.info(f"Saving metrics into {path}...")
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_latest_model(models_dir: Path, pattern: str = "xgb_*.pkl"):
    paths = sorted(models_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No model matching {pattern!r} in {models_dir}")
    return joblib.load(paths[-1]), paths[-1]
