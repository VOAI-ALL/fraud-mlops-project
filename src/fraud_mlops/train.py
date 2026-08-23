from sklearn.linear_model import LogisticRegression

from preprocessing import prepare_data


def train_model():
    # Prepare the dataset
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

    print("Model training completed!")

    return model


if __name__ == "__main__":
    train_model()