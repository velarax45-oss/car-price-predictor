import streamlit as st
import pickle
import pandas as pd

# Load trained pipeline
model = pickle.load(open("car_pipeline.pkl", "rb"))

# Load dataset ONLY for dropdown options
df = pd.read_csv("used_cars_dataset_v2.csv")

st.title("🚗 Car Price Prediction System")

# -----------------------
# INPUT FIELDS
# -----------------------

brand = st.selectbox("Brand", df["Brand"].unique())
car_model = st.selectbox("Model", df["model"].unique())

year = st.number_input("Year", min_value=1990, max_value=2026, value=2015)
age = st.number_input("Age", min_value=0, max_value=40, value=5)

km_driven = st.number_input("KM Driven", min_value=0, value=50000)

transmission = st.selectbox("Transmission", df["Transmission"].unique())
owner = st.selectbox("Owner", df["Owner"].unique())
fuel = st.selectbox("Fuel Type", df["FuelType"].unique())

# -----------------------
# PREDICTION
# -----------------------

if st.button("Predict Price"):

    input_data = pd.DataFrame([[
        year,
        age,
        km_driven,
        brand,
        car_model,
        transmission,
        owner,
        fuel
    ]], columns=[
        "Year",
        "Age",
        "kmDriven",
        "Brand",
        "model",
        "Transmission",
        "Owner",
        "FuelType"
    ])

    prediction = model.predict(input_data)

    st.success(f"Predicted Price: ₹ {int(prediction[0]):,}")