import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('models/model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter transaction details to predict if it's fraud or not.")

# Input fields for 30 features
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
inputs = []

with st.form("fraud_form"):
    for name in feature_names:
        val = st.number_input(f"{name}", format="%.5f")
        inputs.append(val)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Scale and predict
        features_array = np.array(inputs).reshape(1, -1)
        scaled = scaler.transform(features_array)
        prediction = model.predict(scaled)

        # Show result
        result = "🔴 Fraud Detected!" if prediction[0] == 1 else "🟢 Transaction is Safe."
        st.success(result)

    except Exception as e:
        st.error(f"An error occurred: {e}")
