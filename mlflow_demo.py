# mlflow_demo.py
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set tracking URI to SQLite (same as MLflow UI)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Sample data
data = {
    "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
    "salary": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
    "bought": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
}

df = pd.DataFrame(data)

X = df[["age", "salary"]]
y = df["bought"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Start MLflow run
mlflow.set_experiment("xgboost_demo")

with mlflow.start_run():
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_metric("accuracy", acc)

    # Log the model
    mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

    # Save as pickle file
    import pickle
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Accuracy: {acc}")
    print("Model logged to MLflow!")
    print("Model saved to model.pkl!")
