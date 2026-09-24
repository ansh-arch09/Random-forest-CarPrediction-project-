import pandas as pd

# Load Dataset
df = pd.read_csv("data/car_dataset.csv")

print("=" * 50)
print("DATA PREPROCESSING")
print("=" * 50)

# 1. Check Missing Values
print("\n1. Missing Values")
print(df.isnull().sum())

# 2. Check Duplicate Records
print("\n2. Duplicate Records")
print("Total Duplicates :", df.duplicated().sum())

# 3. Check Data Types
print("\n3. Data Types")
print(df.dtypes)

# 4. Remove Duplicates (if any)
df = df.drop_duplicates()

print("\nShape After Removing Duplicates")
print(df.shape)

# 5. Final Check
print("\nMissing Values After Cleaning")
print(df.isnull().sum())

print("\nData Preprocessing Completed Successfully.")