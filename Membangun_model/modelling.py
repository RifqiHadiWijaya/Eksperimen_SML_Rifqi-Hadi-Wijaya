import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


mlflow.set_experiment("Fraud No Tuning")


df = pd.read_csv("Membangun_model/train_fraud.csv")
X = df.drop(columns=["Fraud_Label"])
y = df["Fraud_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

input_example = X_train[0:5]

with mlflow.start_run():
    n_estimators = 100
    random_state = 42
    mlflow.autolog()

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Accuracy:", acc)
