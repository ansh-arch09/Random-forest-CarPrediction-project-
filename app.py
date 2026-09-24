import streamlit as st
import joblib
import pandas as pd

# Load Model
model = joblib.load("models/best_random_forest_model.pkl")

# Load Encoders
car_encoder = joblib.load("models/car_encoder.pkl")
fuel_encoder = joblib.load("models/fuel_encoder.pkl")
seller_encoder = joblib.load("models/seller_encoder.pkl")
transmission_encoder = joblib.load("models/transmission_encoder.pkl")

# Load Dataset
df = pd.read_csv("data/car_dataset_encoded.csv")

st.title("🚗 Car Price Prediction using Random Forest")

st.write("Enter Car Details")

# User Inputs
car_name = st.selectbox("Car Name", sorted(car_encoder.classes_))

year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2025,
    value=2020
)

present_price = st.number_input(
    "Present Price (Lakhs)",
    min_value=0.0,
    value=5.0
)

kms_driven = st.number_input(
    "Kilometers Driven",
    min_value=0,
    value=20000
)

fuel_type = st.selectbox(
    "Fuel Type",
    fuel_encoder.classes_
)

seller_type = st.selectbox(
    "Seller Type",
    seller_encoder.classes_
)

transmission = st.selectbox(
    "Transmission",
    transmission_encoder.classes_
)

owner = st.number_input(
    "Owner",
    min_value=0,
    max_value=5,
    value=0
)

if st.button("Predict Price"):

    car_age = 2025 - year

    mileage_per_year = kms_driven / (car_age + 1)

    premium_brands = [
        "Toyota",
        "Honda",
        "Hyundai",
        "BMW",
        "Audi",
        "Mercedes"
    ]

    premium_brand = 1 if any(
        brand.lower() in car_name.lower()
        for brand in premium_brands
    ) else 0

    median = df["Kms_Driven"].median()

    high_mileage = 1 if kms_driven > median else 0

    new_data = pd.DataFrame({

        "Car_Name":[car_encoder.transform([car_name])[0]],

        "Year":[year],

        "Present_Price":[present_price],

        "Kms_Driven":[kms_driven],

        "Fuel_Type":[fuel_encoder.transform([fuel_type])[0]],

        "Seller_Type":[seller_encoder.transform([seller_type])[0]],

        "Transmission":[transmission_encoder.transform([transmission])[0]],

        "Owner":[owner],

        "Car_Age":[car_age],

        "Mileage_per_Year":[mileage_per_year],

        "Premium_Brand":[premium_brand],

        "High_Mileage":[high_mileage]

    })

    prediction = model.predict(new_data)

    st.success(f"Estimated Selling Price : ₹ {prediction[0]:.2f} Lakhs")