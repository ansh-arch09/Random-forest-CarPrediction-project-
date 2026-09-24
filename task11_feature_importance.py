import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load Model
model = joblib.load("models/best_random_forest_model.pkl")

# Load Dataset
df = pd.read_csv("data/car_dataset_encoded.csv")

# Features
X = df.drop("Selling_Price", axis=1)

# Feature Importance
importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort
feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance\n")
print(feature_df)

# Save CSV
feature_df.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

# Plot Graph
plt.figure(figsize=(10,6))

plt.bar(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.xticks(rotation=45)

plt.title("Random Forest Feature Importance")

plt.xlabel("Features")

plt.ylabel("Importance Score")

plt.tight_layout()

plt.savefig("outputs/feature_importance.png")

plt.show()

print("\nFeature Importance Saved Successfully.")