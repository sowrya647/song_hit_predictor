from flask import Flask, render_template, request
import pandas as pd
import tensorflow as tf
import joblib
import os
from keras.models import load_model

app = Flask(__name__)

# ---------------------------
# Paths & features
# ---------------------------
base_path = os.path.dirname(os.path.abspath(__file__))
features = [
    'danceability', 'energy', 'acousticness', 'valence', 'speechiness',
    'instrumentalness', 'liveness', 'tempo', 'loudness', 'duration_ms'
]

# ---------------------------
# Load model & scaler once
# ---------------------------
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
model = load_model(os.path.join(base_path, "song_popularity_mlp.h5"))

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Get form data
            input_data = {f: float(request.form[f]) for f in features}
            df = pd.DataFrame([input_data])
            
            # Scale and predict
            X_scaled = scaler.transform(df[features])
            pred = (model.predict(X_scaled) > 0.5).astype(int).flatten()[0]
            prediction = "Hit" if pred == 1 else "Non-Hit"
        except Exception as e:
            prediction = f"Error: {e}"
    
    return render_template("index.html", prediction=prediction)

# ---------------------------
# Run app
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
