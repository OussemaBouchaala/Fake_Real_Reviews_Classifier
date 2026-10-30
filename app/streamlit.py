import streamlit as st
import joblib
from pathlib import Path

# Import centralized configuration
from config import REVIEW_PIPELINE
from text_features import predict_label

# -------- Load the trained pipeline --------
# One scikit-learn Pipeline does cleaning, TF-IDF, behavioral features, scaling and the Linear SVM
# (trained in notebooks/pipeline.ipynb), so the app can't drift from the training code.
# config.json paths are relative to the app/ folder
app_dir = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(app_dir / REVIEW_PIPELINE)


model = load_model()

# -------- Streamlit UI --------
st.set_page_config(page_title="Fake Review Detector", layout="centered")

st.title("🕵️ Fake Review Detection (Advanced)")
st.write(
    "This system detects **fake vs real reviews** using machine learning, "
    "combining text meaning **and writing behavior**."
)

user_input = st.text_area("Paste a review below:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    elif predict_label(model, user_input) == "REAL":
        st.success("✅ REAL REVIEW")
    else:
        st.error("❌ FAKE REVIEW")
