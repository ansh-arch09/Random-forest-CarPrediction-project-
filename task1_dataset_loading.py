import pandas as pd

# Load Dataset
df = pd.read_csv("data/car_dataset.csv")

# Display first 5 rows
print("=" * 50)
print("First 5 Rows of Dataset")
print("=" * 50)
print(df.head())

# Shape of Dataset
print("\n" + "=" * 50)
print("Shape of Dataset")
print("=" * 50)
print(df.shape)

# Dataset Information
print("\n" + "=" * 50)
print("Dataset Information")
print("=" * 50)
print(df.info())

# Statistical Summary
print("\n" + "=" * 50)
print("Statistical Summary")
print("=" * 50)
print(df.describe())

# Column Names
print("\n" + "=" * 50)
print("Column Names")
print("=" * 50)
print(df.columns)