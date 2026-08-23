import pandas as pd
import urllib.request
import json
import time


# Load the project dataset
data = pd.read_csv("data/creditcard.csv")


# Use rows 500 to 599 as live/test transactions
test_data = data.iloc[500:600].copy()


# Columns expected by the API
features = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount"
]


# Send 100 predictions
for i, (_, row) in enumerate(test_data.iterrows(), start=1):

    transaction = {
        column: float(row[column])
        for column in features
    }

    request = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=json.dumps(transaction).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(request) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

            print(
                f"{i}/100 -> "
                f"{result['prediction']} | "
                f"probability={result['fraud_probability']}"
            )

    except Exception as error:

        print(f"Prediction {i} failed: {error}")

    time.sleep(0.05)


print("\nFinished sending 100 predictions.")