import os
import joblib
from sklearn.ensemble import RandomForestRegressor

print("=" * 50)
print("RANDOM FOREST MODEL TRAINING")
print("=" * 50)

# Load Training Data
X_train = joblib.load("models/X_train.pkl")
y_train = joblib.load("models/y_train.pkl")

# Create Model
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Model
rf.fit(X_train, y_train)

print("\nModel Training Completed Successfully.")

# Save Model
joblib.dump(rf, "models/random_forest_model.pkl")

print("Random Forest Model Saved Successfully.")