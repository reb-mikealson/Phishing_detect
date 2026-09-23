import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

phi = pd.read_csv("cleaned_data.csv")

shared_cols = ["URLLength", "DomainLength", "IsDomainIP", "NoOfURLRedirect",
               "IsHTTPS", "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL"]
X_phi = phi[shared_cols].copy()

y_phi = 1 - phi["label"]
print("PhiUSIIL shared-feature shape:", X_phi.shape)
print("PhiUSIIL phishing ratio:", y_phi.mean())
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_phi, y_phi)
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

X_new = new_data[shared_cols]
y_new = new_data["phishing"] 

print("\nNew dataset shared-feature shape:", X_new.shape)
print("New dataset phishing ratio:", y_new.mean())


y_pred = model.predict(X_new)

print("\n=== Cross-Dataset Generalization Test ===")
print("Accuracy on NEW dataset:", accuracy_score(y_new, y_pred))
print(classification_report(y_new, y_pred))