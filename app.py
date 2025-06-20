import os
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "supersecretkey"  # for flash messages

# paths to your artifacts
MODEL_PATH      = os.path.join("artifacts", "model.pkl")
VECTORIZER_PATH = os.path.join("artifacts", "preprocessor.pkl")

# load once at startup
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/")
def home():
    # renders home.html in /templates folder
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()
        if not news_text:
            flash("Please enter some text to classify.", "warning")
            return redirect(url_for("predict"))

        # vectorize and predict
        X = vectorizer.transform([news_text])
        pred = model.predict(X)[0]
        label = "Real News ✅" if pred == 1 else "Fake News ❌"

        # render the same index.html with prediction injected
        return render_template("index.html", prediction=label)

    # GET request
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
