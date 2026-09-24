import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

print("=" * 50)
print("HYPERPARAMETER TUNING")
print("=" * 50)

# Create models folder
os.makedirs("models", exist_ok=True)

# Load Training Data
X_train = joblib.load("models/X_train.pkl")
y_train = joblib.load("models/y_train.pkl")

# Base Model
rf = RandomForestRegressor(random_state=42)

# Parameters to Tune
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

# Train Grid Search
grid_search.fit(X_train, y_train)

print("\nBest Parameters Found:")
print(grid_search.best_params_)

print("\nBest R2 Score:")
print(grid_search.best_score_)

# Best Model
best_model = grid_search.best_estimator_

# Save Best Model
joblib.dump(best_model, "models/best_random_forest_model.pkl")

print("\nBest Model Saved Successfully.")

# Save Best Parameters
best_params = pd.DataFrame([grid_search.best_params_])
best_params.to_csv("outputs/best_parameters.csv", index=False)

print("Best Parameters Saved Successfully.")