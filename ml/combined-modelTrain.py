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
