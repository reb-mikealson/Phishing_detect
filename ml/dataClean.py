import pandas as pd
df = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv")
print("Original shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate URLs:", df["URL"].duplicated().sum())
df.drop_duplicates(subset=["URL"], inplace=True)
print("Shape after dropping duplicates:", df.shape)
binary_cols = ["IsDomainIP", "HasObfuscation", "IsHTTPS", "HasTitle",
               "HasFavicon", "Robots", "IsResponsive", "HasDescription",
               "HasExternalFormSubmit", "HasSocialNet", "HasSubmitButton",
               "HasHiddenFields", "HasPasswordField", "Bank", "Pay",
               "Crypto", "HasCopyrightInfo", "label"]
for col in binary_cols:
    print(col,"->",df[col].unique())
print("\nClass distribution (counts):")
print(df["label"].value_counts())
print("\nClass distribution (percentage):")
print(df["label"].value_counts(normalize=True) * 100)
print("\nUnique TLD count:", df["TLD"].nunique())
print("\nTop 20 TLDs:")
print(df["TLD"].value_counts().head(20))
outlier_cols = ["URLLength", "LineOfCode", "LargestLineLength", "NoOfExternalRef"]
print("\nOutlier check (describe):")
print(df[outlier_cols].describe())
print(df.groupby("label")[["URLLength", "LineOfCode", "LargestLineLength", "NoOfExternalRef"]].describe())
corr = df.select_dtypes(include="number").corr()
high_corr = corr[(corr.abs() > 0.9) & (corr != 1.0)]
print(high_corr.stack().drop_duplicates())