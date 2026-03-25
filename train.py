import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set MLflow tracking URI from GitHub Secrets
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

# Create fake dataset (since you don't have real data)
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run() as run:

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    

    # accuracy = 0.7   
    # accuracy = 0.9   # for SUCCESS pipeline

    print(f"Accuracy: {accuracy}")

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Log model (optional but good)
    mlflow.sklearn.log_model(model, "model")

    # Save Run ID to file
    run_id = run.info.run_id
    with open("model_info.txt", "w") as f:
        f.write(run_id)

    print(f"Run ID saved: {run_id}")