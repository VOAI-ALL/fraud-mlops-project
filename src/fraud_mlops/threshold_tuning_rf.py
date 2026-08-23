from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from preprocessing import prepare_data


def main():

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        "data/creditcard.csv"
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Get fraud probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Thresholds to test
    thresholds = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05]

    print("\n===== RANDOM FOREST THRESHOLD TUNING =====")
    print("Threshold | Precision | Recall | F1")
    print("--------------------------------------")

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        print(
            f"{threshold:9.2f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f}"
        )


if __name__ == "__main__":
    main()