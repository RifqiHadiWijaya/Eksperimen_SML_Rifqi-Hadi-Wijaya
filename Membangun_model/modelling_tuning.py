import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Konfigurasi MLflow Lokal
mlflow.set_experiment("Fraud_Hyperparameter_Tuning")

# Load Data
df = pd.read_csv("Membangun_model/train_fraud.csv")
X = df.drop(columns=["Fraud_Label"])
y = df["Fraud_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Definisi Parameter Grid untuk Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Inisialisasi Model Dasar
rf = RandomForestClassifier(class_weight="balanced", random_state=42)

# 3. Hyperparameter Tuning menggunakan GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='f1', # Menggunakan F1 karena data fraud biasanya imbalance
    verbose=2
)

# Menjalankan Tuning
grid_search.fit(X_train, y_train)

# 4. Manual Logging Hasil Terbaik ke MLflow
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

with mlflow.start_run(run_name="Manual_Logging_Tuning"):
    # Log Parameters (Manual)
    mlflow.log_params(best_params)
    mlflow.log_param("model_type", "RandomForest")
    
    # Prediksi
    preds = best_model.predict(X_test)
    
    # Hitung Metriks
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    # Log Metrics (Manual) - Meniru apa yang biasanya dicatat autolog
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # Log Model secara Manual
    mlflow.sklearn.log_model(
        sk_model=best_model, 
        artifact_path="fraud-model",
        input_example=X_train[0:5]
    )
    
    print(f"Tuning Selesai. Best F1: {grid_search.best_score_}")
    print(f"Model tersimpan di MLflow dengan Accuracy: {acc}")