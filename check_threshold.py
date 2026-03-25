import sys
import os

# Read Run ID
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Run ID: {run_id}")

# Read accuracy directly from the mlruns file structure
metrics_path = f"mlruns/0/{run_id}/metrics/accuracy"

if not os.path.exists(metrics_path):
    print(f"ERROR: Metrics file not found at {metrics_path}")
    print("Files in mlruns/0/:")
    for item in os.listdir("mlruns/0"):
        print(f"  {item}")
    sys.exit(1)

with open(metrics_path, "r") as f:
    # MLflow metrics files format: "timestamp value step"
    line = f.read().strip()
    accuracy = float(line.split()[1])

print(f"Accuracy: {accuracy}")

if accuracy < 0.85:
    print("Model did NOT meet threshold")
    sys.exit(1)
else:
    print("Model passed threshold")
