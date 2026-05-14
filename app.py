import streamlit as st
import requests

# Fetch the API URL from Streamlit secrets if deployed, otherwise fallback to localhost
if "API_URL" in st.secrets:
    API_URL = st.secrets["API_URL"]
else:
    API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Ad Click Predictor", page_icon="🎯", layout="centered")

# Inject custom CSS for a premium look
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.2rem;
        font-weight: 800;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(255, 75, 75, 0.2);
    }
    h1, h2, h3 {
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Ad Click Prediction Demo")
st.markdown("Enter the impression details below to get a real-time prediction from the machine learning model.")

with st.container(border=True):
    st.subheader("🌐 Site & App Context")
    st.caption("Provide the anonymized identifiers for the publisher's platform.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        site_id = st.text_input("Site ID", value="1fbe01fe", help="Anonymized ID of the website.")
        app_id = st.text_input("App ID", value="ecad2386", help="Anonymized ID of the mobile application.")
    with col2:
        site_domain = st.text_input("Site Domain", value="f3845767", help="Anonymized domain of the website.")
        app_domain = st.text_input("App Domain", value="7801e8d9", help="Anonymized domain of the mobile application.")
    with col3:
        site_category = st.text_input("Site Category", value="28905ebd", help="Anonymized category of the website.")
        app_category = st.text_input("App Category", value="07d7df22", help="Anonymized category of the mobile application.")

with st.container(border=True):
    st.subheader("📱 Device Information")
    st.caption("Details regarding the user's hardware and connection.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        device_id = st.text_input("Device ID", value="a99f214a", help="Anonymized ID of the user's device.")
        device_type = st.selectbox("Device Type", options=[0, 1, 2, 4, 5], index=1, help="Type of device (e.g., mobile, tablet).")
    with col2:
        device_ip = st.text_input("Device IP", value="eb34399b", help="Anonymized IP address of the device.")
        device_conn_type = st.selectbox("Connection Type", options=[0, 2, 3, 5], index=0, help="Network type (e.g., Wifi, 3G).")
    with col3:
        device_model = st.text_input("Device Model", value="8a4875bd", help="Anonymized model of the device.")

with st.expander("🏷️ Ad & Anonymized Features (C1-C21)", expanded=False):
    st.info("These are hidden categorical features provided by the Ad Exchange to protect proprietary data (e.g., Ad Dimensions, Campaign IDs).")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        banner_pos = st.selectbox("Banner Position", options=[0, 1, 2, 3, 4, 5, 6, 7], index=0, help="Position of the ad banner on the page.")
        c1 = st.number_input("Feature C1", value=1005)
        c14 = st.number_input("Feature C14", value=15701)
    with col2:
        c15 = st.number_input("Feature C15", value=320)
        c16 = st.number_input("Feature C16", value=50)
        c17 = st.number_input("Feature C17", value=2333)
    with col3:
        c18 = st.number_input("Feature C18", value=157)
        c19 = st.number_input("Feature C19", value=35)
        c20 = st.number_input("Feature C20", value=100075)
    with col4:
        c21 = st.number_input("Feature C21", value=23)

st.markdown("---")

if st.button("Predict Click Probability", type="primary"):
    payload = {
        "C1": int(c1),
        "banner_pos": int(banner_pos),
        "site_id": str(site_id),
        "site_domain": str(site_domain),
        "site_category": str(site_category),
        "app_id": str(app_id),
        "app_domain": str(app_domain),
        "app_category": str(app_category),
        "device_id": str(device_id),
        "device_ip": str(device_ip),
        "device_model": str(device_model),
        "device_type": int(device_type),
        "device_conn_type": int(device_conn_type),
        "C14": int(c14),
        "C15": int(c15),
        "C16": int(c16),
        "C17": int(c17),
        "C18": int(c18),
        "C19": int(c19),
        "C20": int(c20),
        "C21": int(c21)
    }

    with st.spinner("Calling API..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                prob = response.json().get("click_probability", 0.5)  # Default to 0.5 if not found
                
                st.markdown("<br>", unsafe_allow_html=True)
                with st.container(border=True):
                    if prob > 0.5:
                        st.success("### 🟢 Likely to Click")
                        st.metric(label="Confidence Score", value=f"{prob:.1%}", delta="High Intent")
                    else:
                        st.error("### 🔴 Unlikely to Click")
                        st.metric(label="Confidence Score", value=f"{prob:.1%}", delta="-Low Intent", delta_color="inverse")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Failed to connect to the API. Is your FastAPI server running?")