from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

from preprocessing import prepare_data


def evaluate_model():

    # Prepare the data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        "data/creditcard.csv"
    )

    # Create the model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Get fraud probabilities
    y_probability = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probability)

    # Display results
    print("\n===== MODEL EVALUATION =====")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # Confusion matrix
    print("\n===== CONFUSION MATRIX =====")
    print(confusion_matrix(y_test, y_pred))

    # Detailed report
    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model()