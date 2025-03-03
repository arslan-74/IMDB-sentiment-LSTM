from flask import Flask, render_template, request
import pickle
import numpy as np
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load Tokenizer & Model
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

model = load_model("sentiment_analysis_LSTM.h5")  # Ensure this file exists

app = Flask(__name__)

def preprocess_text(text):
    """Clean and preprocess input text before prediction."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text.strip()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    
    # Preprocess the input
    cleaned_text = preprocess_text(text)

    # Tokenize & Pad
    seq = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(seq, maxlen=100)  # Adjust `maxlen` based on training

    # Predict
    prediction = model.predict(padded)[0][0]
    
    # Convert to Positive/Negative
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😢"

    return render_template("index.html", result=sentiment)

if __name__ == "__main__":
    app.run(debug=True)
