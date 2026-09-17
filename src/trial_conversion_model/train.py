import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:

    model = XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=8,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
    )
    logger.info(f"Training XGB model on {len(X_train)} rows...")
    model.fit(X_train, y_train)

    return model


def train_lr(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=1000)
    logger.info(f"Training Logistic Regression model on {len(X_train)} rows...")
    lr.fit(X_train_s, y_train)

    return lr


def save_model(model, filename: str, output_dir: Path = Path("models")) -> Path:
    """Save a model as a pickle file.

    Args:
        model: Trained estimator.
        filename: File name without extension (e.g. "xgb" or "xgb-20260916T181200Z").
        output_dir: Destination directory. Defaults to "models" relative to cwd.

    Returns:
        Path of the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.pkl"
    logger.info(f"Saving model into {path}...")
    joblib.dump(model, path)
    return path
