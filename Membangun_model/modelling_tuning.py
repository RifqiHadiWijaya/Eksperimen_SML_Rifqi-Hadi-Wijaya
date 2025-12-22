import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


mlflow.set_experiment("Manual Fraud Tuning")

df = pd.read_csv("Membangun_model/train_fraud.csv")

X = df.drop(columns=["Fraud_Label"])
y = df["Fraud_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10]
}

for n in params["n_estimators"]:
    for d in params["max_depth"]:
        with mlflow.start_run():

            # Train model
            model = RandomForestClassifier(
                n_estimators=n,
                max_depth=d,
                class_weight="balanced",
                random_state=42
            )
            model.fit(X_train, y_train)

            # Prediction
            preds = model.predict(X_test)


            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", d)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            mlflow.sklearn.log_model(model, artifact_path="model")


            with tempfile.TemporaryDirectory() as tmp_dir:

                # -------- Confusion Matrix --------
                cm = confusion_matrix(y_test, preds)

                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")

                cm_path = os.path.join(tmp_dir, "confusion_matrix.png")
                plt.savefig(cm_path)
                plt.close()

                mlflow.log_artifact(cm_path)

                # -------- Prediction File --------
                pred_df = X_test.copy()
                pred_df["y_true"] = y_test.values
                pred_df["y_pred"] = preds

                pred_path = os.path.join(tmp_dir, "predictions.csv")
                pred_df.to_csv(pred_path, index=False)

                mlflow.log_artifact(pred_path)
