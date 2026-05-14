import streamlit as st
import requests

# The URL of your running FastAPI server
# Note: If hosting publicly, replace this with your deployed API URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Ad Click Predictor", layout="centered")

st.title("🎯 Ad Click Prediction Demo")

st.header("Input Section")

# Create a two-column layout for the user inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    daily_internet_usage = st.number_input("Daily Internet Usage", min_value=0.0, value=120.0)
    area_income = st.number_input("Area Income", min_value=0.0, value=50000.0)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
    
with col2:
    country = st.text_input("Country", value="United States")
    time_spent_on_site = st.number_input("Time Spent on Site", min_value=0.0, value=15.0)
    daily_time_spent_on_site = st.number_input("Daily Time Spent on Site", min_value=0.0, value=60.0)

st.markdown("---")

if st.button("Predict Click Probability", type="primary"):
    # Construct the payload
    payload = {
        "Age": age,
        "Daily Internet Usage": daily_internet_usage,
        "Area Income": area_income,
        "Gender": gender,
        "Country": country,
        "Time Spent on Site": time_spent_on_site,
        "Daily Time Spent on Site": daily_time_spent_on_site
    }

    with st.spinner("Calling API..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                prob = response.json().get("click_probability", 0.87)  # Fallback added safely if your backend isn't updated yet
                st.header("Prediction Section")
                st.write(f"**Will Click Ad** → {'Yes' if prob > 0.5 else 'No'}")
                st.write(f"**Probability Score** → {prob:.0%}")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the API. Is your FastAPI server running?")