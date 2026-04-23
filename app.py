import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("car_pipeline.pkl", "rb"))

df = pd.read_csv("used_cars_dataset_v2.csv")

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        🚗 Car Price Prediction App
    </h1>
    <p style='text-align: center; color: grey;'>
        Predict used car prices using Machine Learning
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Enter Car Details")

brand = st.sidebar.selectbox("Brand", df["Brand"].unique())
model_name = st.sidebar.selectbox("Model", df["model"].unique())
year = st.sidebar.number_input("Year", 1990, 2026, 2015)
age = st.sidebar.number_input("Age", 0, 40, 5)
km_driven = st.sidebar.number_input("KM Driven", 0, 500000, 50000)
transmission = st.sidebar.selectbox("Transmission", df["Transmission"].unique())
owner = st.sidebar.selectbox("Owner", df["Owner"].unique())
fuel = st.sidebar.selectbox("Fuel Type", df["FuelType"].unique())

# -----------------------------
# MAIN AREA
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Prediction Section")

    if st.button("🔍 Predict Price"):

        input_data = pd.DataFrame([[
            year,
            age,
            km_driven,
            brand,
            model_name,
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

        prediction = model.predict(input_data)[0]

        st.success("Prediction Completed!")

        st.markdown(
            f"""
            <div style="
                padding: 20px;
                border-radius: 10px;
                background-color: #1e1e1e;
                color: white;
                text-align: center;
                font-size: 22px;
            ">
                Estimated Price: <br>
                <span style="font-size:30px; color:#00ff88;">
                    ₹ {int(prediction):,}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

with col2:
    st.subheader("Info Panel")
    st.info("This model uses Random Forest + preprocessing pipeline")
    st.warning("Prediction is based on historical dataset trends")
    st.write("Tip: Lower KM + newer year = higher price")