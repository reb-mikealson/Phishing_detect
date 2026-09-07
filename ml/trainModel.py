import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


df_model = pd.read_csv("model_ready_data.csv")
print("Loaded shape:", df_model.shape)


top_tlds = df_model["TLD"].value_counts().head(20).index
df_model["TLD"] = df_model["TLD"].apply(lambda x: x if x in top_tlds else "other")
df_model = pd.get_dummies(df_model, columns=["TLD"], drop_first=True)
print("Shape after encoding TLD:", df_model.shape)

df_model = df_model.drop(columns=["URLSimilarityIndex"])
print("Shape after dropping leaky feature:", df_model.shape)

X = df_model.drop(columns=["label"])
y = df_model["label"]


X = df_model.drop(columns=["label"])
y = df_model["label"]
print("Features shape:", X.shape)
print("Target shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

importances = pd.Series(model.feature_importances_, index=X_train.columns)
print("\nTop 15 most important features:")
print(importances.sort_values(ascending=False).head(15))

raw = pd.read_csv("cleaned_data.csv")
print("\nURLSimilarityIndex by class:")
print(raw.groupby("label")["URLSimilarityIndex"].describe())

for col in ["NoOfExternalRef", "NoOfSelfRef", "NoOfCSS", "LineOfCode", "NoOfImage"]:
    print(f"\n{col} by class:")
    print(raw.groupby("label")[col].describe()[["mean", "std", "min", "max"]])

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("Cross-validation scores:", scores)
print("Mean CV accuracy:", scores.mean())

from sklearn.metrics import accuracy_score, classification_report
# Training predictions
y_train_pred = model.predict(X_train)

# Test predictions
y_test_pred = model.predict(X_test)

print("TRAINING PERFORMANCE")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

print("\nTEST PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))