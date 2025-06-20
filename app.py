# app.py

import os
import pickle
import streamlit as st
import streamlit.components.v1 as components

# ——— Load Model & Vectorizer ———
MODEL_PATH      = os.path.join("artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join("artifacts", "preprocessor.pkl")

@st.cache(allow_output_mutation=True)
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        mdl = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vec = pickle.load(f)
    return mdl, vec

model, vectorizer = load_artifacts()

# ——— Helpers to read your HTML templates ———
def load_html_template(filename):
    path = os.path.join("templates", filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

home_html  = load_html_template("home.html")
index_html = load_html_template("index.html")

# ——— Page setup ———
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")
page = st.sidebar.radio("Go to", ["Home", "Predict"])

if page == "Home":
    # Embed your home.html (static) inside Streamlit
    components.html(
        home_html,
        height=400,  # adjust to fit
        scrolling=True
    )

else:  # Predict page
    # First, embed the static portion of index.html up through your <form> tag
    # We'll strip out the form itself, since Streamlit will render it
    split_at = index_html.split("<form")[0]  
    components.html(split_at, height=200, scrolling=False)

    # Now use Streamlit widget in place of the form
    news_text = st.text_area("Enter News Text", height=250)
    if st.button("Predict"):
        if not news_text.strip():
            st.warning("Please enter some text to classify.")
        else:
            X = vectorizer.transform([news_text])
            pred = model.predict(X)[0]
            label = "Real News ✅" if pred == 1 else "Fake News ❌"
            # Finally, render the “prediction alert” area from your index.html,
            # substituting {{ prediction }} with the real label:
            alert_html = f"""
            <div class="alert alert-info text-center mt-4" role="alert">
              <h4 class="alert-heading">Prediction Result</h4>
              <p class="mb-0"><strong>{label}</strong></p>
            </div>
            """
            components.html(alert_html, height=120)

