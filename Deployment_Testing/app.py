
import streamlit as st
import pickle
import numpy as np

# Title of the application
st.title("Driver Alertness Detection")

# Sidebar for user input
st.sidebar.header("User Input Features")

def user_input_features():
    physiological = st.sidebar.slider("Physiological Signal", 0, 100, 50)
    environmental = st.sidebar.slider("Environmental Factor", 0, 100, 50)
    vehicle_data = st.sidebar.slider("Vehicle Data", 0, 100, 50)

    data = {
        'physiological': physiological,
        'environmental': environmental,
        'vehicle_data': vehicle_data
    }
    return data

input_data = user_input_features()

# Convert user input to a DataFrame
import pandas as pd
input_df = pd.DataFrame([input_data])

st.write("### Input Data:", input_df)

# Load the pre-trained model
model_file = 'driver_alertness_model.pkl'
try:
    model = pickle.load(open(model_file, 'rb'))
    
    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display prediction
    st.subheader("Prediction")
    result = "Alert" if prediction[0] == 1 else "Not Alert"
    st.write(result)

    st.subheader("Prediction Probability")
    st.write(prediction_proba)

except FileNotFoundError:
    st.error(f"Model file '{model_file}' not found. Please ensure the model is available.")
