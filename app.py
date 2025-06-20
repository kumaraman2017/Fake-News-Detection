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
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

# -------------------------------
# Load HTML templates
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
# Session state navigation
# -------------------------------

if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar selector
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Predict"],
    index=["Home", "Predict"].index(st.session_state.page)
)
st.session_state.page = page

# -------------------------------
# Apply custom CSS for theme-aware colors
# -------------------------------

st.markdown("""
    <style>
    :root {
        color-scheme: light dark;
    }
    h1, h2, h3, h4, p, label {
        color: inherit !important;
    }
    .custom-alert {
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .custom-success {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid rgba(0, 255, 0, 0.4);
    }
    .custom-warning {
        background-color: rgba(255, 255, 0, 0.1);
        border: 1px solid rgba(255, 255, 0, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# HOME PAGE
# -------------------------------

if st.session_state.page == "Home":
    components.html(home_html, height=600, scrolling=True)

    if st.button("➡️ Go to Predictor"):
        st.session_state.page = "Predict"
        st.experimental_rerun()

# -------------------------------
# PREDICT PAGE
# -------------------------------

elif st.session_state.page == "Predict":
    static_part = index_html.split("<form")[0]
    components.html(static_part, height=150, scrolling=False)

    news_text = st.text_area("Enter News Text", height=200)

    if st.button("Predict"):
        if not news_text.strip():
            st.warning("⚠️ Please enter some text.")
        else:
            X = vectorizer.transform([news_text])
            pred = model.predict(X)[0]
            label = "✅ Real News" if pred == 1 else "❌ Fake News"

            st.markdown(
                f"""
                <div class="custom-alert custom-success">
                    <h4>Prediction Result</h4>
                    <p style="font-size: 1.5rem;"><strong>{label}</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )

    if st.button("⬅️ Back to Home"):
        st.session_state.page = "Home"
        st.experimental_rerun()
