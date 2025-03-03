import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Ensure CPU usage only
os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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

# Load Tokenizer at Startup
try:
    with open("tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)

    # Load TensorFlow Lite Model
    interpreter = tf.lite.Interpreter(model_path="sentiment_analysis_LSTM.tflite")
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    logging.info("✅ Tokenizer and TFLite model loaded successfully.")

except Exception as e:
    logging.error(f"⚠️ Error loading tokenizer or model: {e}")
    raise SystemExit("Failed to load tokenizer or model.")

def preprocess_text(text):
    """Clean and preprocess input text before prediction."""
    try:
        text = text.lower().strip()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
        text = re.sub(r"\d+", "", text)  # Remove numbers
        text = re.sub(r"\s+", " ", text)  # Normalize spaces
        text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
        return text
    except Exception as e:
        logging.error(f"⚠️ Text preprocessing error: {e}")
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
        padded = pad_sequences(seq, maxlen=100, dtype='float32')

        # Run inference on TensorFlow Lite model
        interpreter.set_tensor(input_details[0]['index'], padded)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        # Determine sentiment
        sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😢"

        return sentiment  
        # jsonify({"sentiment": sentiment, "confidence": float(prediction)})

    except Exception as e:
        logging.error(f"⚠️ Prediction error: {e}")
        return jsonify({"error": "An error occurred during prediction."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logging.info(f"🚀 Starting Flask app on port {port}...")
    serve(app, host="0.0.0.0", port=port)
