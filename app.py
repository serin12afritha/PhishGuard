import streamlit as st
import joblib
import pandas as pd
import ipaddress
from urllib.parse import urlparse

st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ PhishGuard")
st.subheader("Phishing Website Detection System")

st.write(
    "Enter a website URL to predict whether "
    "it is potentially phishing or legitimate."
)

# Load URL-based model
@st.cache_resource
def load_model():
    return joblib.load("phishguard_url_model.pkl")


# Extract the same 12 features used during training
def extract_url_features(url):
    url = str(url).strip()

    parsed = urlparse(
        url if "://" in url else "http://" + url
    )

    hostname = parsed.hostname or ""
    path = parsed.path or ""

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "count_dots": url.count("."),
        "count_hyphens": url.count("-"),
        "count_at": url.count("@"),
        "count_question": url.count("?"),
        "count_equal": url.count("="),
        "count_digits": sum(c.isdigit() for c in url),
        "count_slashes": url.count("/"),
        "has_https": int(parsed.scheme == "https"),
        "has_ip": has_ip
    }


try:
    model = load_model()

    st.success("Machine Learning model loaded successfully!")

    st.header("Website URL Analysis")

    website_url = st.text_input(
        "Enter Website URL",
        placeholder="https://example.com"
    )

    if st.button("Check Website", type="primary"):

        if not website_url.strip():
            st.warning("Please enter a website URL.")

        else:
            parsed = urlparse(
                website_url if "://" in website_url
                else "http://" + website_url
            )

            if not parsed.hostname or "." not in parsed.hostname:
                st.error("Please enter a valid website URL.")

            else:
                features = extract_url_features(website_url)

                input_data = pd.DataFrame([features])

                if hasattr(model, "feature_names_in_"):
                    input_data = input_data[
                        list(model.feature_names_in_)
                    ]

                prediction = model.predict(input_data)[0]

                st.subheader("Prediction Result")

                if prediction == 1:
                    st.error("⚠️ Potential Phishing URL Detected!")
                else:
                    st.success("✅ URL Classified as Legitimate")

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_data)[0]
                    phishing_index = list(model.classes_).index(1)

                    st.write(
                        "Estimated phishing probability:",
                        f"{probabilities[phishing_index]:.2%}"
                    )

                st.caption(
                    "This prediction is based only on URL characteristics. "
                    "It does not verify the website's actual content or safety."
                )

except Exception as e:
    st.error("Unable to load the machine learning model.")
    st.exception(e)
