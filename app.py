import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import re
import string
import logging
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from waitress import serve  # Use Waitress for production

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Load Tokenizer & Model at Startup
try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)

    model = tf.keras.models.load_model("sentiment_analysis_LSTM.h5")
    logging.info("Model and tokenizer loaded successfully.")

except Exception as e:
    logging.error(f"Error loading model/tokenizer: {e}")
    raise SystemExit("Failed to load model or tokenizer.")

def preprocess_text(text):
    """Clean and preprocess input text before prediction."""
    try:
        text = text.lower().strip()  # Convert to lowercase & strip spaces
        text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"\s+", " ", text)  # Normalize spaces
        text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        return text
    except Exception as e:
        logging.error(f"Text preprocessing error: {e}")
        return ""

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text", "").strip()

        if not text:
            return jsonify({"error": "Please enter valid text."}), 400

        cleaned_text = preprocess_text(text)

        # Tokenize & Pad
        seq = tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=100)  # Ensure `maxlen` matches training

        # Predict
        prediction = model.predict(padded)[0][0]
        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😢"

        return jsonify({"sentiment": sentiment, "confidence": float(prediction)})

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == "__main__":
    logging.info("Starting Flask app...")
    serve(app, host="0.0.0.0", port=5000)  # Use Waitress for production
