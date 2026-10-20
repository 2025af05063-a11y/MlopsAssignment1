import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_heart_data(path="data/raw/processed.cleveland.data"):
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","target"]
    df = pd.read_csv(path, names=cols, na_values="?")
    # Binarize target: 0 = no disease, 1-4 = disease
    df["target"] = (df["target"] > 0).astype(int)
    # Drop rows with missing (only 6 rows in cleveland)
    df = df.dropna()
    return df

def build_preprocessor():
    numeric = ["age","trestbps","chol","thalach","oldpeak"]
    categorical = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", "passthrough", categorical)  # already encoded as integers
    ])