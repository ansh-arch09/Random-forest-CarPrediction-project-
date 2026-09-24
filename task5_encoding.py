import pandas as pd
import os
import joblib
from sklearn.preprocessing import LabelEncoder

# Load Featured Dataset
df = pd.read_csv("data/car_dataset_featured.csv")

print("=" * 50)
print("LABEL ENCODING")
print("=" * 50)

# Create models folder
os.makedirs("models", exist_ok=True)

# Create Encoders
car_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()
seller_encoder = LabelEncoder()
transmission_encoder = LabelEncoder()

# Encode Columns
df["Car_Name"] = car_encoder.fit_transform(df["Car_Name"])
df["Fuel_Type"] = fuel_encoder.fit_transform(df["Fuel_Type"])
df["Seller_Type"] = seller_encoder.fit_transform(df["Seller_Type"])
df["Transmission"] = transmission_encoder.fit_transform(df["Transmission"])

# Save Encoders
joblib.dump(car_encoder, "models/car_encoder.pkl")
joblib.dump(fuel_encoder, "models/fuel_encoder.pkl")
joblib.dump(seller_encoder, "models/seller_encoder.pkl")
joblib.dump(transmission_encoder, "models/transmission_encoder.pkl")

# Save Encoded Dataset
df.to_csv("data/car_dataset_encoded.csv", index=False)

print("\nEncoding Completed Successfully.\n")
print(df.head())

print("\nEncoders Saved Successfully.")
print("Encoded Dataset Saved Successfully.")