import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# Load Encoded Dataset
df = pd.read_csv("data/car_dataset_encoded.csv")

print("=" * 50)
print("TRAIN TEST SPLIT")
print("=" * 50)

# Features (X)
X = df.drop("Selling_Price", axis=1)

# Target (y)
y = df["Selling_Price"]

print("\nFeatures Shape :", X.shape)
print("Target Shape :", y.shape)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape")
print(X_train.shape)

print("\nTesting Data Shape")
print(X_test.shape)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save Split Data
joblib.dump(X_train, "models/X_train.pkl")
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("\nTrain-Test Split Completed Successfully.")
print("Files Saved Successfully.")