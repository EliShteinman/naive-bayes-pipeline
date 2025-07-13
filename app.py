# app.py

import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.title("🍄 Naive Bayes Mushroom Classifier")

# --- שלב 1: קבלת הסכמה ---
@st.cache_data
def fetch_schema():
    res = requests.get(f"{API_BASE}/expected-features")
    return res.json()

schema = fetch_schema()

# --- שלב 2: בניית טופס דינמי ---
with st.form("prediction_form"):
    user_input = {}
    for feature, options in schema.items():
        user_input[feature] = st.selectbox(f"{feature}", options)
    submitted = st.form_submit_button("Predict")

# --- שלב 3: שליחת קלט וקבלת תחזית ---
if submitted:
    response = requests.post(
        f"{API_BASE}/predict",
        json={"features": user_input}
    )
    if response.status_code == 200:
        result = response.json()
        st.success(f"Prediction: {result['prediction']}")
    else:
        st.error(f"Error: {response.json()['detail']}")