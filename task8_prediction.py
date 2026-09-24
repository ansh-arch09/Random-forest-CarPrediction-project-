import os
import joblib
import pandas as pd

print("=" * 50)
print("MODEL PREDICTION")
print("=" * 50)

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load Model
rf = joblib.load("models/random_forest_model.pkl")

# Load Test Data
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

# Prediction
predictions = rf.predict(X_test)

# Create DataFrame
result = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

# Save Prediction File
result.to_csv("outputs/predictions.csv", index=False)

print("\nFirst 10 Predictions\n")
print(result.head(10))

print("\nPrediction Completed Successfully.")
print("Predictions saved in outputs/predictions.csv")