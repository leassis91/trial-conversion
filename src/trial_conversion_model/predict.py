import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def predict(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:

    probs = model.predict_proba(X_test)[:, 1]

    auc = round(roc_auc_score(y_test, probs), 4)

    return {"AUC": auc}


def save_metrics(metrics: dict, output_dir: Path, run_id: str, n_test: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "n_test": n_test, "metrics": metrics}
    path = output_dir / f"metrics-{run_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
