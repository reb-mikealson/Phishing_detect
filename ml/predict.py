import joblib
import pandas as pd
from urllib.parse import urlparse

# Load the trained model
model = joblib.load("phishing_model.pkl")

def extract_features(url):
    parsed = urlparse(url)
    domain = parsed.netloc

    features = {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "IsDomainIP": 1 if domain.replace(".", "").isdigit() else 0,
        "NoOfURLRedirect": 0,   # can't know this without visiting the site
        "IsHTTPS": 1 if parsed.scheme == "https" else 0,
        "NoOfEqualsInURL": url.count("="),
        "NoOfQMarkInURL": url.count("?"),
        "NoOfAmpersandInURL": url.count("&"),
    }
    return pd.DataFrame([features])

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

# ---- Try it out ----
if __name__ == "__main__":
    predict_url("https://www.google.com")
    predict_url("http://secure-login-update-account.xyz/verify?user=123&token=456")
    predict_url("https://www.wikipedia.org")