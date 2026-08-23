from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response
from collections import deque
from src.fraud_mlops.drift.detector import detect_drift
import joblib
import pandas as pd
import yaml


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Real-Time Credit Card Fraud Detection API",
    description="Machine learning API for credit card fraud detection",
    version="1.0.0"
)


# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------

model = joblib.load("models/fraud_model.joblib")
scaler = joblib.load("models/scaler.joblib")

with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

THRESHOLD = params["model"]["threshold"]


# --------------------------------------------------
# Load reference data for drift monitoring
# --------------------------------------------------

reference_data = pd.read_csv(
    "data/creditcard.csv"
).iloc[:500].copy()


# --------------------------------------------------
# Prometheus metrics
# --------------------------------------------------

prediction_counter = Counter(
    "fraud_predictions_total",
    "Total number of predictions",
    ["prediction"]
)

prediction_latency = Histogram(
    "fraud_prediction_latency_seconds",
    "Prediction latency in seconds"
)

data_drift_detected = Gauge(
    "data_drift_detected",
    "Whether data drift has been detected"
)


# --------------------------------------------------
# Current data window for drift monitoring
# --------------------------------------------------

current_data_window = deque(maxlen=100)


# --------------------------------------------------
# Transaction input schema
# --------------------------------------------------

class Transaction(BaseModel):

    Time: float

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float


# --------------------------------------------------
# Home endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Fraud Detection API is running"
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "Random Forest"
    }


# --------------------------------------------------
# Prometheus metrics endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():

    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(transaction: Transaction):

    # Convert transaction into dictionary
    transaction_data = transaction.model_dump()

    # Convert dictionary into DataFrame
    data = pd.DataFrame(
        [transaction_data],
        columns=[
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
    )

    # --------------------------------------------------
    # Apply scaler
    # --------------------------------------------------

    data_scaled = scaler.transform(data)


    # --------------------------------------------------
    # Get fraud probability
    # --------------------------------------------------

    with prediction_latency.time():

        probability = model.predict_proba(
            data_scaled
        )[0][1]


    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    if probability >= THRESHOLD:

        prediction = "FRAUD"

    else:

        prediction = "NORMAL"


    # --------------------------------------------------
    # Update prediction Prometheus metric
    # --------------------------------------------------

    prediction_counter.labels(
        prediction=prediction
    ).inc()


    # --------------------------------------------------
    # Add current transaction to drift window
    # --------------------------------------------------

    current_data_window.append(
        transaction_data
    )


    # --------------------------------------------------
    # Drift detection
    # --------------------------------------------------

    if len(current_data_window) == 100:

        current_data = pd.DataFrame(
            list(current_data_window)
        )

        drift_detected, drift_results = detect_drift(
            reference_data,
            current_data
        )

        if drift_detected:

            data_drift_detected.set(1)

        else:

            data_drift_detected.set(0)


    # --------------------------------------------------
    # Return prediction
    # --------------------------------------------------

    return {
        "prediction": prediction,
        "fraud_probability": round(
            float(probability),
            4
        ),
        "threshold": THRESHOLD
    }