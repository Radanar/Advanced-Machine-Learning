import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_map = joblib.load("cluster_labels.pkl")  # Load saved models

st.title("Customer Segment Predictor")

recency = st.number_input("Recency (days)", min_value=0)
frequency = st.number_input("Frequency", min_value=0)
monetary = st.number_input("Monetary Value", min_value=0.0)

# If the button is clicked, check with the labels from trained model
if st.button("Predict Segment"):
    input_data = np.array([[recency, frequency, monetary]])
    input_scaled = scaler.transform(input_data)
    segment = model.predict(input_scaled)[0]
    label = label_map.get(segment, "Unknown Segment")

    st.success(f"The predicted customer segment is: {label}")


