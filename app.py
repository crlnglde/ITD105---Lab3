import streamlit as st
import numpy as np
import joblib

# Page Title
st.title("ML Model Prediction App")
st.write("Upload a trained model, provide inputs, and make predictions.")

# Step 1: Select Task Type

st.sidebar.title("Navigation")
task_type = st.sidebar.radio("Choose Task", ["Classification (Heart Disease)", "Regression (Soil Moisture Content)"])

# Function for user input in Heart Disease model (Classification)
def get_heart_disease_input():
    st.subheader("Heart Disease Prediction Inputs")
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    chest_pain_type = st.selectbox("Chest Pain Type (1-4)", [1, 2, 3, 4])
    bp = st.number_input("BP (80-200)", min_value=80, max_value=200, value=80)
    cholesterol = st.number_input("Cholesterol (100-600)", min_value=100, max_value=600, value=100)
    fbs_over_120 = st.selectbox("FBS > 120 (0: No, 1: Yes)", [0, 1])
    ekg_results = st.selectbox("EKG Results (0-2)", [0, 1, 2])
    max_hr = st.number_input("Max HR (60-220)", min_value=60, max_value=220, value=60)
    exercise_angina = st.selectbox("Exercise Angina (0: No, 1: Yes)", [0, 1])
    st_depression = st.number_input("ST Depression (0.0-6.0)", min_value=0.0, max_value=6.0, value=0.0)
    slope_of_st = st.selectbox("Slope of ST (1-3)", [1, 2, 3])
    number_of_vessels_fluro = st.selectbox("Number of Vessels Fluoro (0-3)", [0, 1, 2, 3])
    thallium = st.selectbox("Thallium (3, 6, 7)", [3, 6, 7])

    features = np.array([[age, sex, chest_pain_type, bp, cholesterol, fbs_over_120, 
                          ekg_results, max_hr, exercise_angina, st_depression, 
                          slope_of_st, number_of_vessels_fluro, thallium]])
    return features

# Function for user input in Soil Moisture model (Regression)
def get_soil_moisture_input():
    st.subheader("Soil Moisture Content Prediction Inputs")
    pH = st.number_input("pH", min_value=0.0, value=10.5)
    organic_matter_percent = st.number_input("Organic Matter Percent", min_value=0.0, value=3.5)
    nitrogen_percent = st.number_input("Nitrogen Percent", min_value=0.0, value=0.1)
    phosphorus_ppm = st.number_input("Phosphorus ppm", min_value=0, value=15)
    potassium_ppm = st.number_input("Potassium ppm", min_value=0, value=200)

    features = np.array([[pH, organic_matter_percent, nitrogen_percent, phosphorus_ppm, potassium_ppm]])
    return features

# Classification Section
if task_type == "Classification (Heart Disease)":
    st.subheader("Classification: Heart Disease Prediction")

    # Upload trained model
    model_file = st.file_uploader("Upload a trained classification model", type="pkl")
    if model_file is not None:
        model = joblib.load(model_file)
        st.success("Classification model loaded successfully.")

        # Get user input
        features = get_heart_disease_input()

        # Predict
        if st.button("Predict Heart Disease"):
            prediction = model.predict(features)[0]
            prediction_label = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
            st.write(f"**Prediction:** {prediction_label}")

# Regression Section
elif task_type == "Regression (Soil Moisture Content)":
    st.subheader("Regression: Soil Moisture Content Prediction")

    # Upload trained model
    model_file = st.file_uploader("Upload a trained regression model", type="pkl")
    if model_file is not None:
        model = joblib.load(model_file)
        st.success("Regression model loaded successfully.")

        # Get user input
        features = get_soil_moisture_input()

        # Predict
        if st.button("Predict Soil Moisture Content"):
            prediction = model.predict(features)[0]
            st.write(f"**Predicted Soil Moisture Content:** {prediction:.2f}")
            
            # Provide insights based on the predicted moisture content percent
            if prediction < 10:
                st.write("**Insight:** Indicates dry soil conditions.")
            elif 10 <= prediction < 21:
                st.write("**Insight:** Soil is adequately moist.")
            elif 21 <= prediction < 31:
                st.write("**Insight:** Ideal for most plants.")
            elif 31 <= prediction < 41:
                st.write("**Insight:** Soil is quite moist, bordering on wet.")
            else:   #prediction >= 41
                st.write("**Insight:** Soil is saturated.")
