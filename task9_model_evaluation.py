import os
import joblib
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from math import sqrt

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load Model
rf = joblib.load("models/random_forest_model.pkl")

# Load Test Data
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

# Prediction
y_pred = rf.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = sqrt(mse)

r2 = r2_score(y_test, y_pred)

# Print Results
print(f"\nMean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Save Evaluation Report
report = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R2 Score"],
    "Value": [mae, mse, rmse, r2]
})

report.to_csv("outputs/evaluation_report.csv", index=False)

print("\nEvaluation Report Saved Successfully.")