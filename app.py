import os
import pickle
import streamlit as st
import streamlit.components.v1 as components

# -------------------------------
# Load model & vectorizer (cached)
# -------------------------------

MODEL_PATH = os.path.join("artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join("artifacts", "preprocessor.pkl")

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        mdl = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vec = pickle.load(f)
    return mdl, vec

model, vectorizer = load_artifacts()

# -------------------------------
# Load HTML templates
# -------------------------------

def load_html_template(filename):
    path = os.path.join("templates", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

home_html = load_html_template("home.html")
index_html = load_html_template("index.html")

# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

# -------------------------------
# Use session_state for navigation
# -------------------------------

if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Sidebar OR in-page navigation
with st.sidebar:
    st.session_state.page = st.radio("Go to", ["Home", "Predict"], index=["Home", "Predict"].index(st.session_state.page))

# -------------------------------
# HOME PAGE
# -------------------------------

if st.session_state.page == "Home":
    components.html(
        home_html,
        height=500,
        scrolling=True
    )

    # Add a Streamlit button to navigate
    if st.button("Check News Now"):
        st.session_state.page = "Predict"
        st.experimental_rerun()

# -------------------------------
# PREDICT PAGE
# -------------------------------

else:
    split_at = index_html.split("<form")[0]
    components.html(split_at, height=200, scrolling=False)

    news_text = st.text_area("Enter News Text", height=250)

    if st.button("Predict"):
        if not news_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            X = vectorizer.transform([news_text])
            pred = model.predict(X)[0]
            label = "✅ Real News" if pred == 1 else "❌ Fake News"

            st.markdown(
                f"""
                <div style="text-align: center; margin-top: 1rem;">
                  <h4 style="color: var(--text-color);">Prediction Result</h4>
                  <p style="font-size: 1.5rem; color: var(--text-color);"><strong>{label}</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
