import os
import pickle
import streamlit as st
import streamlit.components.v1 as components

# -------------------------------
# Load model & vectorizer
# -------------------------------

MODEL_PATH = os.path.join("artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join("artifacts", "preprocessor.pkl")

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# -------------------------------
# Load HTML parts
# -------------------------------

def load_html(filename):
    with open(os.path.join("templates", filename), "r", encoding="utf-8") as f:
        return f.read()

home_html = load_html("home.html")
index_html = load_html("index.html")

# -------------------------------
# Streamlit configuration
# -------------------------------

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# -------------------------------
# Page selection in sidebar
# -------------------------------

page = st.sidebar.radio("Go to", ["Home", "Predict"])

# -------------------------------
# HOME PAGE
# -------------------------------

if page == "Home":
    components.html(home_html, height=500, scrolling=False)

    if st.button("Check News Now"):
        st.session_state['GoTo'] = "Predict"
        st.experimental_rerun()

# -------------------------------
# PREDICT PAGE
# -------------------------------

else:
    components.html(index_html, height=120, scrolling=False)

    news_text = st.text_area("Enter News Text", height=200)

    if st.button("Predict"):
        if not news_text.strip():
            st.warning("Please enter some text.")
        else:
            X = vectorizer.transform([news_text])
            pred = model.predict(X)[0]
            label = "✅ Real News" if pred == 1 else "❌ Fake News"
            st.success(f"**Prediction:** {label}")

    if st.button("Back to Home"):
        st.session_state['GoTo'] = "Home"
        st.experimental_rerun()

# -------------------------------
# Optional: handle jump via session_state
# -------------------------------

if 'GoTo' in st.session_state:
    if st.session_state['GoTo'] != page:
        page = st.session_state['GoTo']
        st.session_state.pop('GoTo')
        st.experimental_rerun()
