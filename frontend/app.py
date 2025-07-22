# frontend/app.py
"""
A Streamlit web application to interact with the Mushroom Classifier API.

This app dynamically fetches the model's expected features from the backend,
builds a user input form, sends the user's data for prediction, and displays
the result.
"""
import os
import requests
import streamlit as st

# --- 1. Configuration ---
# Use an environment variable for the API URL, with a fallback for local development.
# This makes the app more flexible for different deployment scenarios.
API_BASE_URL = os.getenv("API_BASE_URL", "http://host.docker.internal:8000")

st.set_page_config(page_title="Mushroom Classifier", page_icon="🍄")
st.title("🍄 Naive Bayes Mushroom Classifier")


# --- 2. Data Fetching ---
@st.cache_data(show_spinner="Fetching model features...")
def fetch_schema() -> dict | None:
    """
    Fetches the expected input schema from the backend's /expected-features endpoint.

    Caches the result to avoid repeated API calls.

    Returns:
        dict | None: A dictionary representing the schema, or None if an error occurs.
    """
    try:
        url = f"{API_BASE_URL}/expected-features"
        response = requests.get(url, timeout=5)  # Added timeout for robustness
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the backend API: {e}")
        return None


schema = fetch_schema()

# --- 3. UI and Prediction Logic ---
if not schema:
    st.warning(
        "Could not load model features from the backend. The service might be unavailable. Please try again later.")
else:
    with st.form("prediction_form"):
        st.write("Please provide the mushroom's features:")
        user_input = {}
        # Dynamically create a selectbox for each feature
        for feature, options in schema.items():
            user_input[feature] = st.selectbox(label=feature.replace("-", " ").title(), options=options)

        submitted = st.form_submit_button("Predict")

    if submitted:
        predict_url = f"{API_BASE_URL}/predict"
        try:
            # The backend now expects a GET request with query parameters
            response = requests.get(predict_url, params=user_input, timeout=5)
            response.raise_for_status()

            result = response.json()
            prediction = result.get("prediction")

            if prediction == "p":
                st.error(f"Prediction: Poisonous 🤢")
            elif prediction == "e":
                st.success(f"Prediction: Edible 😋")
            else:
                st.info(f"Prediction: {prediction}")

        except requests.exceptions.HTTPError as e:
            # Handle specific API errors (e.g., 400 Bad Request)
            error_detail = e.response.json().get("detail", "No details provided.")
            st.error(f"API Error: {error_detail} (Status code: {e.response.status_code})")
        except requests.exceptions.RequestException as e:
            # Handle network errors
            st.error(f"Failed to get a prediction from the backend: {e}")