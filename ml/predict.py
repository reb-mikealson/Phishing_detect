import joblib
import pandas as pd
from urllib.parse import urlparse

# Load the trained model
model = joblib.load("phishing_model.pkl")

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
def predict_url(url):
    features = extract_features(url)
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    label = "PHISHING" if prediction == 1 else "LEGITIMATE"
    confidence = probability[prediction] * 100

    print(f"\nURL: {url}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    return label