# app.py

import streamlit as st
import pickle
import os

# Load model and vectorizer
MODEL_PATH = os.path.join("artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join("artifacts", "preprocessor.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Enter your news text below to check whether it's **Real** or **Fake**.")

news_text = st.text_area("News Text", height=200)

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Vectorize & predict
        transformed_text = vectorizer.transform([news_text])
        pred = model.predict(transformed_text)[0]
        label = "✅ Real News" if pred == 1 else "❌ Fake News"

        st.success(f"**Prediction:** {label}")
