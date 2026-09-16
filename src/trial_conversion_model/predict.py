import pandas as pd


def predict(model, X_test: pd.DataFrame) -> pd.Series:

    probs = model.predict_proba(X_test)[:, 1]

    return probs
