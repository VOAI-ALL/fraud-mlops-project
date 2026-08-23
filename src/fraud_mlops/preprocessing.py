import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load the credit card dataset."""
    return pd.read_csv(file_path)


def prepare_data(file_path):
    """Prepare data for machine learning."""

    df = load_data(file_path)

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("Training fraud cases:", y_train.sum())
    print("Testing fraud cases:", y_test.sum())

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler
    )


if __name__ == "__main__":
    prepare_data("data/creditcard.csv")