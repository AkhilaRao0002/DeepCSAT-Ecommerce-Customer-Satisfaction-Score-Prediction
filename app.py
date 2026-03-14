import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =============================
# Load saved files
# =============================
model = load_model("csat_model.keras")
scaler = joblib.load("scaler.pkl")
training_columns = joblib.load("training_columns.pkl")

# =============================
# Streamlit UI
# =============================
st.title("Customer Satisfaction (CSAT) Prediction")

st.write("Enter the support interaction details to predict CSAT score.")

# User Inputs
response_time = st.number_input("Response Time", min_value=0.0)
category = st.text_input("Category")
sub_category = st.text_input("Sub Category")
agent_name = st.text_input("Agent Name")
channel = st.text_input("Channel Name")
agent_shift = st.text_input("Agent Shift")
manager = st.text_input("Manager")
tenure = st.text_input("Tenure Bucket")

# =============================
# Prediction
# =============================
if st.button("Predict CSAT Score"):

    # Create dataframe
    sample = pd.DataFrame({
        'Response_Time':[response_time],
        'category':[category],
        'Sub-category':[sub_category],
        'Agent_name':[agent_name],
        'channel_name':[channel],
        'Agent Shift':[agent_shift],
        'Manager':[manager],
        'Tenure Bucket':[tenure]
    })

    # One-hot encode
    sample = pd.get_dummies(sample)

    # Align columns with training data
    sample = sample.reindex(columns=training_columns, fill_value=0)

    # Scale
    sample_scaled = scaler.transform(sample)

    # Predict
    pred_prob = model.predict(sample_scaled)
    pred_class = np.argmax(pred_prob, axis=1)

    # Convert back to CSAT scale
    csat_score = pred_class[0] + 1

    st.success(f"Predicted CSAT Score: {csat_score}")
