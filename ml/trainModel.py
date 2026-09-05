import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
df_model = pd.read_csv("model_ready_data.csv")
print("Loaded shape:", df_model.shape)
X = df_model.drop(columns=["label"])
y = df_model["label"]
print("Features shape:", X.shape)
print("Target shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
