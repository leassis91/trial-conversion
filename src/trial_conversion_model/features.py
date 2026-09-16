import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data.

    Args:
        df (pd.DataFrame): Dataframe raw.

    Returns:
        pd.DataFrame: Processed dataframe ready for split.
    """
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["trial_started_at"] = pd.to_datetime(df["trial_started_at"])

    for col in ["day1_share", "listen_share", "avg_session_minutes"]:
        df[col] = df[col].fillna(0)

    return df


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["sessions_3d"] = data[["sessions_day1", "sessions_day2", "sessions_day3"]].sum(
        axis=1
    )
    data["active_days_3d"] = (
        data[["sessions_day1", "sessions_day2", "sessions_day3"]] > 0
    ).sum(axis=1)
    data["day1_share"] = data["sessions_day1"] / data["sessions_3d"]
    data["listen_share"] = data["listen_sessions_3d"] / data["sessions_3d"]
    data["avg_session_minutes"] = data["total_minutes_3d"] / data["sessions_3d"]

    return data


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature engineering of preprocessed dataframe.

    Args:
        df (pd.DataFrame): Processed dataframe.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Dataframe
    """
    # IDCOL = "trial_id"
    TARGET = "converted"
    # FEATURES = [col for col in df.columns if col not in [TARGET, IDCOL]]

    df = add_features(df)
    df = preprocess_data(df)

    FEATURES_FILTERED = [
        "sessions_3d",
        "active_days_3d",
        "day1_share",
        "listen_share",
        "avg_session_minutes",
        "total_minutes_3d",
        "country",
        "device_type",
    ]

    X = pd.get_dummies(df[FEATURES_FILTERED], columns=["country", "device_type"])
    y = df[TARGET]

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test
