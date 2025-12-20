import streamlit as st
import joblib
import re
import string
import numpy as np
from nltk.corpus import stopwords
from scipy.sparse import hstack

# -------- Load model & vectorizer --------
tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("linear_svm_model_adv.pkl")

# -------- Stopwords --------
stop_words = set(stopwords.words("english"))

# -------- Text cleaning (for TF-IDF) --------
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -------- Behavioral feature extraction (RAW text) --------
def extract_behavior_features(text):
    words = text.split()
    return np.array([
        len(text),                                  # length
        len(words),                                 # 2. word count
        text.count("!"),                            # exclamations
        sum(1 for c in text if c.isupper()) / max(len(text), 1),  # capital ratio
        len(words) / max(len(set(words)), 1)        # repetition ratio
    ]).reshape(1, -1)

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
    else:
        # TF-IDF features
        cleaned = clean_text(user_input)
        X_text = tfidf.transform([cleaned])

        # Behavioral features
        X_behav = extract_behavior_features(user_input)

        # Combine
        X_final = hstack([X_text, X_behav])

        # Predict
        prediction = model.predict(X_final)

        if prediction[0] == 1:
            st.success("✅ REAL REVIEW")
        else:
            st.error("❌ FAKE REVIEW")
