import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ PhishGuard")
st.subheader("Phishing Website Detection System")

st.write(
    "PhishGuard uses Machine Learning "
    "to identify phishing and legitimate websites."
)

@st.cache_resource
def load_model():
    return joblib.load("phishguard_model.pkl")

model = load_model()

st.success("Machine Learning model loaded successfully!")

st.info(
    "The model is ready. We will add website feature "
    "input and prediction in the next step."
)
