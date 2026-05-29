import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def load_data(path: str) -> pd.DataFrame:
    """Load sample data."""
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic feature engineering: one-hot encode categorical variables."""
    feature_cols = ["age", "education_years", "industry", "job_type", "employment_stability"]
    X = pd.get_dummies(df[feature_cols], drop_first=True)
    return X


def train_predictor(df: pd.DataFrame, target_col: str = "target_metric"):
    """Train a prediction model on labeled samples and predict all records."""
    labeled = df[df["is_labeled"]].copy()
    X_labeled = build_features(labeled)
    y = labeled[target_col]

    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf=5)
    model.fit(X_labeled, y)

    X_all = build_features(df)
    X_all = X_all.reindex(columns=X_labeled.columns, fill_value=0)

    df = df.copy()
    df["pred_metric"] = model.predict(X_all)
    return df, model


def calibrated_mean(df: pd.DataFrame, target_col: str = "target_metric", pred_col: str = "pred_metric"):
    """Full-sample predicted mean plus residual correction from labeled samples."""
    labeled = df[df["is_labeled"]]
    direct_prediction = df[pred_col].mean()
    correction = (labeled[target_col] - labeled[pred_col]).mean()
    return direct_prediction + correction


def evaluate(df: pd.DataFrame):
    labeled = df[df["is_labeled"]]
    true_mean = df["target_metric"].mean()
    sample_mean = labeled["target_metric"].mean()
    direct_mean = df["pred_metric"].mean()
    calibrated = calibrated_mean(df)

    return pd.DataFrame({
        "method": ["sample_mean", "direct_prediction", "calibrated_estimation"],
        "estimate": [sample_mean, direct_mean, calibrated],
        "bias_vs_full_data": [sample_mean - true_mean, direct_mean - true_mean, calibrated - true_mean]
    })
