import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# ---- Load PhiUSIIL data ----
phi = pd.read_csv("cleaned_data.csv")

shared_cols = ["URLLength", "DomainLength", "IsDomainIP", "NoOfURLRedirect",
               "IsHTTPS", "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL"]

X_phi = phi[shared_cols].copy()
y_phi = 1 - phi["label"]  # flip so 1 = phishing (matches new dataset)

new_data = pd.read_csv("dataset_cybersecurity_michelle.csv")

rename_map = {
    "length_url": "URLLength",
    "domain_length": "DomainLength",
    "domain_in_ip": "IsDomainIP",
    "qty_redirects": "NoOfURLRedirect",
    "tls_ssl_certificate": "IsHTTPS",
    "qty_equal_url": "NoOfEqualsInURL",
    "qty_questionmark_url": "NoOfQMarkInURL",
    "qty_and_url": "NoOfAmpersandInURL",
}
new_data = new_data.rename(columns=rename_map)

X_new = new_data[shared_cols].copy()
y_new = new_data["phishing"]

# ---- Combine both datasets ----
X_combined = pd.concat([X_phi, X_new], ignore_index=True)
y_combined = pd.concat([y_phi, y_new], ignore_index=True)

print("Combined shape:", X_combined.shape)
print("Combined phishing ratio:", y_combined.mean())

# ---- Train/test split on the COMBINED data ----
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, stratify=y_combined, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n=== Combined Dataset — Test Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# ---- Cross-validation for stability ----
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_combined, y_combined, cv=cv, scoring="accuracy", n_jobs=-1)
print("\nShuffled CV scores:", scores)
print("Mean CV accuracy:", scores.mean())

# ---- THE REAL TEST: train on PhiUSIIL only, test on new dataset only (repeat for comparison) ----
model_phi_only = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_phi_only.fit(X_phi, y_phi)
y_pred_cross = model_phi_only.predict(X_new)
print("\n=== For comparison: PhiUSIIL-only model tested on new dataset ===")
print("Accuracy:", accuracy_score(y_new, y_pred_cross))

# ---- NEW TEST: train on COMBINED, but evaluate separately on each original dataset ----
model_combined = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_combined.fit(X_train, y_train)

# Evaluate specifically on new-dataset-only rows within the test set
print("\n=== Combined-trained model, evaluated on New dataset alone ===")
pred_new_only = model_combined.predict(X_new)
print("Accuracy:", accuracy_score(y_new, pred_new_only))

print("\n=== Combined-trained model, evaluated on PhiUSIIL alone ===")
pred_phi_only = model_combined.predict(X_phi)
print("Accuracy:", accuracy_score(y_phi, pred_phi_only))

import joblib
joblib.dump(model_combined, "phishing_model.pkl")
print("Model saved as phishing_model.pkl")