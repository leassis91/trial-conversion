import logging
from pathlib import Path

import joblib
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def load_model(filename):
    model_path = Path(__file__).parent / f"models/{filename}.pkl"
    model = joblib.load(model_path)

    return model


def evaluate_model(y_test, probs):

    auc = round(roc_auc_score(y_test, probs), 4)
    logger.info(f"XGBoost AUC: {auc:.4f)}")

    return {"AUC": auc}


if __name__ == "__main__":
    load_model()
    evaluate_model()
