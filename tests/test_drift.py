import pandas as pd

from fraud_mlops.drift.detector import detect_drift


# Load your project dataset
data = pd.read_csv("data/creditcard.csv")

# Use the first 500 rows as reference data
reference_data = data.iloc[:500].copy()

# Use the next 500 rows as current/live data
current_data = data.iloc[500:1000].copy()

# Detect drift
drift_detected, drift_results = detect_drift(
    reference_data,
    current_data
)

print("Drift detected:", drift_detected)

print("\nDrift results:")

for feature, score in drift_results.items():
    print(f"{feature}: {score:.4f}")