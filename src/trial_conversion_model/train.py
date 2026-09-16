from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


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

    model.fit(X_train, y_train)

    return model


def train_lr(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_s, y_train)

    return lr


def save_model(model, filename: str):
    """Save model in a pickle file.

    Args:
        model (_type_): _description_
        name_file (str): _description_
    """
    output_path = Path(__file__).parent / f"models/{filename}.pkl"
    joblib.dump(model, output_path)
