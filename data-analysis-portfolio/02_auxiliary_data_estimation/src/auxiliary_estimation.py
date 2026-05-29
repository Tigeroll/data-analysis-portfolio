import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def check_response_bias(df: pd.DataFrame):
    """Compare respondents with full-sample records."""
    return pd.DataFrame({
        "group": ["full_data", "responded_sample"],
        "x_score_mean": [df["x_score"].mean(), df.loc[df["is_responded"], "x_score"].mean()],
        "auxiliary_score_mean": [df["auxiliary_score"].mean(), df.loc[df["is_responded"], "auxiliary_score"].mean()]
    })


def predict_and_calibrate(df: pd.DataFrame):
    """Train a predictor on responded samples and correct prediction bias with residuals."""
    train = df[df["is_responded"]].copy()
    X_train = train[["x_score", "auxiliary_score"]]
    y_train = train["true_response"]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    df = df.copy()
    df["pred_response"] = model.predict(df[["x_score", "auxiliary_score"]])
    correction = (train["true_response"] - df.loc[df["is_responded"], "pred_response"]).mean()
    calibrated_estimate = df["pred_response"].mean() + correction

    return calibrated_estimate, df, model
