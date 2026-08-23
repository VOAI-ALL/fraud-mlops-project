import joblib

from sklearn.ensemble import RandomForestClassifier

from preprocessing import prepare_data


def train_and_save_model():

    print("Preparing data...")

    X_train, X_test, y_train, y_test, scaler = prepare_data(
        "data/creditcard.csv"
    )

    print("Training Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Save model
    joblib.dump(
        model,
        "models/fraud_model.joblib"
    )

    # Save scaler
    joblib.dump(
        scaler,
        "models/scaler.joblib"
    )

    print("\nModel saved successfully!")
    print("models/fraud_model.joblib")

    print("\nScaler saved successfully!")
    print("models/scaler.joblib")


if __name__ == "__main__":
    train_and_save_model()