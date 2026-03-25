import mlflow
import sys

# Read Run ID
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

# Get MLflow tracking URI from environment
mlflow.set_tracking_uri("file:./mlruns")

# Fetch run
run = mlflow.get_run(run_id)

accuracy = run.data.metrics.get("accuracy", 0)

print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")

# Check threshold
if accuracy < 0.85:
    print("Model did NOT meet threshold ")
    sys.exit(1)
else:
    print("Model passed threshold ")
