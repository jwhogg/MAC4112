from abc import ABC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
import polars as pl
import plotly.express as px
import os
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
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class Model(ABC):
    def __init__(self, name: str, estimator, **kwargs):
        self.name = name
        self.estimator = estimator(**kwargs)
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.estimator.fit(X_train, y_train)
        return self
    def predict(self, X_test) -> np.ndarray:
        return self.estimator.predict(X_test)
def evaluate(model: Model, X_test: np.ndarray, y_test: np.ndarray, labels=None):
    print(f"{model.name}:\n")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig = px.imshow(
        cm, x=labels, y=labels, text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        title=f"Confusion Matrix — {model.name}",
    )
    fig.write_image(os.path.join(OUTPUT_DIR, f"confusion_matrix_{model.name.replace(' ', '_')}.png"))
    return y_pred, report
def main():
    DATA_PATH = "code/gold_data/gold_layer.parquet"
    df = pl.read_parquet(DATA_PATH)
    exclude_cols = ["trial", "run", "fault_mode", "routine"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df.select(feature_cols).to_numpy()
    y = df["fault_mode"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.35, stratify=y, random_state=42
    )
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    LR_model = Model("Logistic Regression", LogisticRegression)
    XGB_model = Model("XGBoost", xgb.XGBClassifier)
    RF_model = Model("Random Forest", RandomForestClassifier, n_estimators=300, random_state=42)
    SVC_model = Model("SVC", SVC, kernel="rbf", C=1.0)
    KNN_model = Model("KNN", KNeighborsClassifier, n_neighbors=5)
    LR_model.fit(X_train, y_train)
    XGB_model.fit(X_train, y_train_enc)
    RF_model.fit(X_train, y_train)
    SVC_model.fit(X_train, y_train)
    KNN_model.fit(X_train, y_train)
    summary_rows = []
    _, report = evaluate(LR_model, X_test, y_test, labels=le.classes_)
    summary_rows.append({"model": LR_model.name, "accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]})
    _, report = evaluate(XGB_model, X_test, y_test_enc, labels=np.unique(y_test_enc))
    summary_rows.append({"model": XGB_model.name, "accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]})
    _, report = evaluate(RF_model, X_test, y_test, labels=le.classes_)
    summary_rows.append({"model": RF_model.name, "accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]})
    _, report = evaluate(SVC_model, X_test, y_test, labels=le.classes_)
    summary_rows.append({"model": SVC_model.name, "accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]})
    _, report = evaluate(KNN_model, X_test, y_test, labels=le.classes_)
    summary_rows.append({"model": KNN_model.name, "accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]})

    summary_df = pl.DataFrame(summary_rows)
    heatmap_values = summary_df.select(["accuracy", "macro_f1"]).to_numpy()
    heatmap_fig = px.imshow(
        heatmap_values,
        x=["accuracy", "macro_f1"],
        y=summary_df["model"].to_list(),
        text_auto=".3f",
        color_continuous_scale="RdYlGn",
        title="Model Comparison Heatmap",
    )
    heatmap_fig.write_image(os.path.join(OUTPUT_DIR, "model_comparison_heatmap.png"))

if __name__ == "__main__":
    main()
