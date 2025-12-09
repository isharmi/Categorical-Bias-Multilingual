import pandas as pd
import numpy as np

# Load WALS values
values = pd.read_csv("/Users/sharmiislam/Documents/Paper Work/Linguistic Typology & Bias Study/data/values.csv")
codes = pd.read_csv("/Users/sharmiislam/Documents/Paper Work/Linguistic Typology & Bias Study/data/codes.csv")

# Clean + merge
values = values.rename(columns={'Language_ID':'LanguageID', 'Parameter_ID':'FeatureID', 'Value':'Value'})
codes = codes.rename(columns={'ID':'CodeID','Name':'Category'})
df = values.merge(codes, left_on='Code_ID', right_on='CodeID', how='left')

# Pivot feature matrix
feature_matrix = df.pivot(index='LanguageID', columns='FeatureID', values='Category')
feature_matrix = feature_matrix.fillna("missing")

# Encode
from sklearn.preprocessing import LabelEncoder
for col in feature_matrix.columns:
    feature_matrix[col] = LabelEncoder().fit_transform(feature_matrix[col])

X = feature_matrix.values.astype(float)

# Compute bias score = distance to mean language
mean_vec = X.mean(axis=0)
bias_scores = np.linalg.norm(X - mean_vec, axis=1)

# Save bias_scores.csv
bias_df = pd.DataFrame({
    "LanguageID": feature_matrix.index,
    "BiasScore": bias_scores
})

bias_df.to_csv("bias_scores.csv", index=False)
print("Saved bias_scores.csv")
