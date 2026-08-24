
import os

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from preprocessing import prepare_data


# --------------------------------------------------
# MLflow configuration
# --------------------------------------------------

# IMPORTANT:
# Airflow is running inside Docker.
# Therefore MLflow must use a Linux/container path,
# NOT the Windows I:\... path.
MLFLOW_TRACKING_URI = "file:///opt/airflow/mlruns"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def train_model():

    # --------------------------------------------------
    # Prepare dataset
    # --------------------------------------------------

    X_train, X_test, y_train, y_test, scaler = prepare_data(
        "data/creditcard.csv"
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    print(f"Training fraud cases: {sum(y_train)}")
    print(f"Testing fraud cases: {sum(y_test)}")

    # --------------------------------------------------
    # MLflow experiment
    # --------------------------------------------------

    mlflow.set_experiment("Fraud Detection")

    # --------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------

    with mlflow.start_run():

        # --------------------------------------------------
        # Model parameters
        # --------------------------------------------------

        max_iter = 1000
        random_state = 42

        # --------------------------------------------------
        # Create model
        # --------------------------------------------------

        model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
        )

        # --------------------------------------------------
        # Train model
        # --------------------------------------------------

        model.fit(X_train, y_train)

        # --------------------------------------------------
        # Predictions
        # --------------------------------------------------

        y_pred = model.predict(X_test)

        # --------------------------------------------------
        # Calculate metrics
        # --------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        # --------------------------------------------------
        # Log parameters to MLflow
        # --------------------------------------------------

        mlflow.log_param(
            "model",
            "LogisticRegression",
        )

        mlflow.log_param(
            "max_iter",
            max_iter,
        )

        mlflow.log_param(
            "random_state",
            random_state,
        )

        # --------------------------------------------------
        # Log metrics to MLflow
        # --------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "f1_score",
            f1,
        )

        # --------------------------------------------------
        # Log model to MLflow
        # --------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            "model",
        )

        # --------------------------------------------------
        # Display results
        # --------------------------------------------------

        print()
        print("======================================")
        print("Model training completed successfully")
        print("======================================")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print()
        print("MLflow Tracking URI:")
        print(MLFLOW_TRACKING_URI)

        print()
        print("MLflow Run ID:")
        print(mlflow.active_run().info.run_id)

        return model


if __name__ == "__main__":
    train_model()
