import os
import pickle
import streamlit as st
import streamlit.components.v1 as components

# ——— Load artifacts with cache_resource ———
MODEL_PATH      = os.path.join("artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join("artifacts", "preprocessor.pkl")

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        mdl = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vec = pickle.load(f)
    return mdl, vec

model, vectorizer = load_artifacts()

# ——— Helpers ———
def load_html(filename):
    with open(os.path.join("templates", filename), "r", encoding="utf-8") as f:
        return f.read()

home_html  = load_html("home.html")
index_html = load_html("index.html")

# ——— Determine current page via query param “page” ———
params = st.query_params()
page = params.get("page", ["Home"])[0]  # default to Home

# ——— Render ———
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

if page == "Home":
    # Inject a link that updates the query param to ?page=Predict
    # Note: Clicking it reloads the app with page="Predict"
    linked_home = home_html.replace(
        'href="/predict"',
        'href="?page=Predict"'
    )
    components.html(linked_home, height=400, scrolling=True)

elif page == "Predict":
    # Embed up to <form> from index.html (static header)
    prefix = index_html.split("<form")[0]
    # Also replace form action so it doesn’t try to POST anywhere
    prefix = prefix.replace('action="/predict"', '')
    components.html(prefix, height=200, scrolling=False)

    # Streamlit widget for the text area & predict button
    news_text = st.text_area("Enter News Text", height=250)
    if st.button("Predict"):
        if not news_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            X = vectorizer.transform([news_text])
            pred = model.predict(X)[0]
            label = "✅ Real News" if pred == 1 else "❌ Fake News"

            # Inject the Bootstrap alert from index.html
            alert_html = f"""
            <div class="alert alert-info text-center mt-4" role="alert">
              <h4 class="alert-heading">Prediction Result</h4>
              <p class="mb-0"><strong>{label}</strong></p>
            </div>
            """
            components.html(alert_html, height=120)

else:
    st.error("Unknown page!")

