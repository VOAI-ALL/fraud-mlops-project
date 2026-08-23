from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

from preprocessing import prepare_data


def evaluate_thresholds():

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        "data/creditcard.csv"
    )

    # Train balanced model
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Get probability of fraud
    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [0.50, 0.30, 0.20, 0.10, 0.05]

    print("\n===== THRESHOLD COMPARISON =====")
    print("Threshold | Precision | Recall | F1")
    print("--------------------------------------")

    for threshold in thresholds:

        # Convert probabilities into predictions
        y_pred = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        print(
            f"{threshold:9.2f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f}"
        )


if __name__ == "__main__":
    evaluate_thresholds()