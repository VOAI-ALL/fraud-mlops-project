import pandas as pd

# Load the dataset
df = pd.read_csv("data/creditcard.csv")

# Display basic information
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["Class"].value_counts())

fraud_percentage = (
    df["Class"].value_counts(normalize=True)[1] * 100
)

print("\nFraud percentage:", fraud_percentage, "%")