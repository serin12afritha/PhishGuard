
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
    "A Machine Learning-based system "
    "to identify phishing and legitimate websites."
)

@st.cache_resource
def load_model():
    return joblib.load("phishguard_model.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("dataset_phishing.csv")

try:
    model = load_model()
    df = load_dataset()

    st.success("Machine Learning model loaded successfully!")

    st.header("Website Prediction")

    st.write(
        "Choose a sample website from the dataset "
        "to test the prediction."
    )

    row_number = st.slider(
        "Select website sample",
        min_value=0,
        max_value=len(df) - 1,
        value=0
    )

    website = df.iloc[row_number]

    st.write("**Website URL:**")
    st.code(str(website["url"]))

    if st.button("Check Website", type="primary"):

        X = df.drop(columns=["url", "status"])

        input_data = X.iloc[[row_number]]

        if hasattr(model, "feature_names_in_"):
            input_data = input_data[
                list(model.feature_names_in_)
            ]

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Phishing Website Detected!")
        else:
            st.success("✅ Legitimate Website")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            predicted_index = list(model.classes_).index(prediction)
            probability = probabilities[predicted_index]

            st.write(
                f"Model-estimated probability: {probability:.2%}"
            )

except Exception as e:
    st.error("Something went wrong while loading the application.")
    st.exception(e)
