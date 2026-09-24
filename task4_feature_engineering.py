import pandas as pd

# Load Dataset
df = pd.read_csv("data/car_dataset.csv")

print("=" * 50)
print("FEATURE ENGINEERING")
print("=" * 50)

# -----------------------------------
# Feature 1 : Car Age
# -----------------------------------

CURRENT_YEAR = 2025

df["Car_Age"] = CURRENT_YEAR - df["Year"]

# -----------------------------------
# Feature 2 : Mileage Per Year
# -----------------------------------

df["Mileage_per_Year"] = df["Kms_Driven"] / (df["Car_Age"] + 1)

# -----------------------------------
# Feature 3 : Premium Brand
# -----------------------------------

premium_brands = [
    "Toyota",
    "Honda",
    "Hyundai",
    "BMW",
    "Audi",
    "Mercedes"
]

df["Premium_Brand"] = df["Car_Name"].apply(
    lambda x: 1 if any(
        brand.lower() in x.lower()
        for brand in premium_brands
    ) else 0
)

# -----------------------------------
# Feature 4 : High Mileage
# -----------------------------------

median = df["Kms_Driven"].median()

df["High_Mileage"] = df["Kms_Driven"].apply(
    lambda x: 1 if x > median else 0
)

print("\nNew Features Added Successfully.\n")

print(df.head())

# Save Updated Dataset
df.to_csv("data/car_dataset_featured.csv", index=False)

print("\nDataset Saved Successfully")