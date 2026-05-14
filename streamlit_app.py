import streamlit as st
import requests

# The URL of your running FastAPI server
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Ad Click Predictor", layout="centered")

st.title("🎯 Ad Click Prediction Demo")
st.write("This frontend connects to our production FastAPI server to get real-time click probabilities.")

# Create a nice two-column layout for the user inputs
col1, col2 = st.columns(2)

with col1:
    site_domain = st.text_input("Site Domain", value="f3845767", help="Anonymized site domain string")
    device_type = st.selectbox("Device Type", options=[0, 1, 4, 5], index=1, help="Type of device (e.g., mobile, tablet)")
    
with col2:
    banner_pos = st.selectbox("Banner Position", options=[0, 1, 2, 3, 4, 5, 6, 7], index=0)
    c14 = st.number_input("C14 (Ad Campaign ID)", value=15701)

st.markdown("---")

# The predict button
if st.button("Predict Click Probability", type="primary"):
    # Construct the payload matching your AdClickPayload Pydantic model
    payload = {
        "C1": 1005,
        "banner_pos": banner_pos,
        "site_id": "1fbe01fe",
        "site_domain": site_domain,
        "site_category": "28905ebd",
        "app_id": "ecad2386",
        "app_domain": "7801e8d9",
        "app_category": "07d7df22",
        "device_ip": "eb34399b",
        "device_model": "8a4875bd",
        "device_type": device_type,
        "device_conn_type": 0,
        "C14": c14,
        "C15": 320,
        "C16": 50,
        "C17": 2333,
        "C18": 157,
        "C19": 35,
        "C20": 100075,
        "C21": 23
    }

    with st.spinner("Calling API..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                prob = response.json()["click_probability"]
                
                # Display the result beautifully
                st.metric(label="Predicted Click Probability", value=f"{prob:.2%}")
                
                if prob > 0.5:
                    st.success("High intent! Good candidate for bidding.")
                else:
                    st.warning("Low intent. Might want to skip this bid.")
                    
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the API. Is your FastAPI server running?")