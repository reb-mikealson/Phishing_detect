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
