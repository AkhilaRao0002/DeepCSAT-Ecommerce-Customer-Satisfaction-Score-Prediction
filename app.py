import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =============================
# Load saved files
# =============================
model = load_model("csat_model")
scaler = joblib.load("scaler.pkl")
training_columns = joblib.load("training_columns.pkl")

# =============================
# Streamlit UI
# =============================
st.title("Customer Satisfaction (CSAT) Prediction")

st.write("Enter ticket details to predict CSAT score")

# User Inputs
response_time = st.number_input("Response Time", min_value=0)

category = st.selectbox(
    "Category",
    ["Order related","Technical","Payment","Account"]
)

sub_category = st.selectbox(
    "Sub-category",
    ["Delayed","Refund","Login issue","Other"]
)

agent = st.text_input("Agent Name", "Stanley Hogan")

channel = st.selectbox(
    "Channel Name",
    ["Inbound","Email","Chat"]
)

shift = st.selectbox(
    "Agent Shift",
    ["Morning","Split","Afternoon"]
)

manager = st.text_input("Manager", "Emily Chen")

tenure = st.selectbox(
    "Tenure Bucket",
    ["0-30","30-60","60-90",">90"]
)

# =============================
# Prediction Button
# =============================
if st.button("Predict CSAT Score"):

    # Create dataframe
    sample = pd.DataFrame({
        'Response_Time':[response_time],
        'category':[category],
        'Sub-category':[sub_category],
        'Agent_name':[agent],
        'channel_name':[channel],
        'Agent Shift':[shift],
        'Manager':[manager],
        'Tenure Bucket':[tenure]
    })

    # Convert categorical variables
    sample = pd.get_dummies(sample)

    # Match training columns
    sample = sample.reindex(columns=training_columns, fill_value=0)

    # Scale input
    sample_scaled = scaler.transform(sample)

    # Predict
    pred_prob = model.predict(sample_scaled)
    pred_class = np.argmax(pred_prob, axis=1)

    # Convert back to CSAT scale
    pred_csat = pred_class + 1

    st.success(f"Predicted CSAT Score: {pred_csat[0]}")
