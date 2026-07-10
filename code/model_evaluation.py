from abc import ABC
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import numpy as np
import polars as pl

"""
This Code uses the gold layer dataset, and evaluates a number of models for detecting defects
"""

"""
Model class usage:
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb

    lr_model = Model("Logistic Regression", LogisticRegression, max_iter=2000)
    xgb_model = Model("XGBoost", xgb.XGBClassifier, n_estimators=200, max_depth=4, eval_metric="mlogloss")

    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
"""

class Model(ABC):
    def __init__(self, name: str, estimator, **kwargs):
        self.name = name
        self.estimator = estimator(**kwargs)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.estimator.fit(X_train, y_train)
        return self

    def predict(self, X_test) -> np.ndarray:
        return self.estimator.predict(X_test)


def evaluate(model: Model, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return y_pred


def main():

    DATA_PATH = "gold_data/gold_layer.parquet_2026-07-08_12-07-25"

    df = pl.read_parquet(self.gold_data_path)

    exclude_cols = ["trial", "run", "fault_mode", "routine"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df.select(feature_cols).to_numpy()
    y = df["fault_mode"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, stratify=y, random_state=42
    )

    LR_model = Model("Logistic Regression", LogisticRegression)
    XGB_model = Model("XGBoost", xgb.XGBClassifier)

    LR_model.fit(X_train, y_train)


if __name__ == "__main__":
    main()

# need to have a generic model class that is capable of taking data and producing predictions
# whatever models that are to be tested will use this generic class
# will have a seperate evaluation function
