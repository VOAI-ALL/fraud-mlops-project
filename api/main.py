from fastapi import FastAPI
from pydantic import BaseModel
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

    # Apply the same scaler used during training
    data_scaled = scaler.transform(data)

    # Get fraud probability
    probability = model.predict_proba(data_scaled)[0][1]

    # Classification threshold
    if probability >= THRESHOLD:
        prediction = "FRAUD"
    else:
        prediction = "NORMAL"

    return {
    "prediction": prediction,
    "fraud_probability": round(float(probability), 4),
    "threshold": THRESHOLD
}