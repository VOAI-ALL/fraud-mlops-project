import pandas as pd


REQUIRED_COLUMNS = [
    "Time",
    "Amount",
    "Class"
]


def validate_dataset(file_path):
    print("Loading dataset...")

    df = pd.read_csv(file_path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Check required columns
    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Required columns: OK")

    # Check missing values
    missing_values = df.isnull().sum().sum()

    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values} missing values"
        )

    print("Missing values: OK")

    # Check target values
    valid_classes = set(df["Class"].unique())

    if not valid_classes.issubset({0, 1}):
        raise ValueError(
            f"Invalid Class values: {valid_classes}"
        )

    print("Target values: OK")

    print("Dataset validation successful!")

    return df


if __name__ == "__main__":
    validate_dataset("data/creditcard.csv")