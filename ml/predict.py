import joblib
import pandas as pd
from urllib.parse import urlparse

# Load the trained model
model = joblib.load("phishing_model.pkl")

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc
