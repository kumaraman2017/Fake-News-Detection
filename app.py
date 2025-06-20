import os
import pickle
import streamlit as st

# ----------------------------------------
# Load artifacts
# ----------------------------------------

MODEL_PATH = os.path.join("artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join("artifacts", "preprocessor.pkl")

@st.cache_resource
def load_model_and_vectorizer():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# ----------------------------------------
# Page config
# ----------------------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
)

# ----------------------------------------
# Sidebar navigation
# ----------------------------------------

if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Sidebar selector
choice = st.sidebar.radio("Navigate", ["Home", "Predict"])

# ----------------------------------------
# HOME PAGE
# ----------------------------------------

if choice == "Home":
    st.markdown("""
        <div style='text-align: center;'>
            <h1>🚀 Fake News Detection</h1>
            <p>Check if a news article is Real or Fake using Machine Learning.</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Check News Now"):
        st.session_state.page = "Predict"
        st.experimental_rerun()

# ----------------------------------------
# PREDICT PAGE
# ----------------------------------------

elif choice == "Predict":
    st.markdown("""
        <div style='text-align: center;'>
            <h1>📰 Fake News Predictor</h1>
        </div>
    """, unsafe_allow_html=True)

    news_text = st.text_area("Enter News Text Below:", height=200)

    if st.button("Predict"):
        if not news_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            X = vectorizer.transform([news_text])
            pred = model.predict(X)[0]
            label = "✅ Real News" if pred == 1 else "❌ Fake News"
            st.success(f"**Prediction:** {label}")

    if st.button("Back to Home"):
        st.session_state.page = "Home"
        st.experimental_rerun()
