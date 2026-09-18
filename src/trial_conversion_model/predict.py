import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Returns the AUC of a classification model, given a `X_test` pandas dataset and a `y_test` pandas' series.

    Args:
        model: Any fitted classifier exposing `predict_proba`.
        X_test (pd.DataFrame): Held-out features, one row per sample.
        y_test (pd.Series): Held-out binary labels, aligned with `X_test`.

    Returns:
        dict: {"AUC": `roc_auc_score`}
    """

    probs = model.predict_proba(X_test)[:, 1]
    logger.info(f"Evaluating model {str(model)[:3]} on {len(y_test)} rows...")
    auc = round(roc_auc_score(y_test, probs), 4)
    logger.info(f"Metrics: {auc}")

    return {"AUC": auc}


def save_metrics(metrics: dict, output_dir: Path, run_id: str) -> Path:
    """Save model evaluation's metrics in `models` directory.

    Args:
        metrics (dict): Metric name to value, as returned by `evaluate`
            (e.g. {"AUC": 0.83}).
        output_dir (Path): Directory the JSON file is written to. Created
            with its parents if it does not exist.
        run_id (str): Identifier for this run. Used in the filename and
            stored inside the payload.

    Returns:
        Path: Path to the JSON file just written,
            `output_dir/metrics-<run_id>.json`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "metrics": metrics}
    path = output_dir / f"metrics-{run_id}.json"
    logger.info(f"Saving metrics into {path}...")
    path.write_text(json.dumps(payload, indent=2))
    return path


def load_metrics(metrics_dir: Path) -> pd.DataFrame:
    """Load every metrics JSON in a directory as a comparison table.

    Reads the files written by `save_metrics`, in filename order.

    Args:
        metrics_dir (Path): Directory holding the `metrics-<run_id>.json`
            files. Files not matching that pattern are ignored.

    Returns:
        pd.DataFrame: One row per metrics file, with the nested payload
            flattened into columns (`run_id`, `metrics.AUC`, ...). Empty
            if the directory holds no matching file.
    """
    return pd.json_normalize(
        [json.loads(p.read_text()) for p in sorted(metrics_dir.glob("metrics-*.json"))]
    )


def load_latest_model(models_dir: Path, pattern: str = "xgb_*.pkl"):
    paths = sorted(models_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No model matching {pattern!r} in {models_dir}")
    return joblib.load(paths[-1]), paths[-1]
