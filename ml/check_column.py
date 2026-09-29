import joblib
model = joblib.load("phishing_model_full.pkl")
print(list(model.feature_names_in_))