import pandas as pd
df = pd.read_csv("data/PhiUSIIL_Phishing_URL_Dataset.csv")
id_cols = ["FILENAME", "URL", "Domain", "Title"]
df_model = df.drop(columns=id_cols)
print("Shape after removing duplicates:", df.shape)