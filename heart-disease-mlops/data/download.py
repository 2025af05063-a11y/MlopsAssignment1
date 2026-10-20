import urllib.request
import os

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMNS = ["age","sex","cp","trestbps","chol","fbs","restecg",
           "thalach","exang","oldpeak","slope","ca","thal","target"]

os.makedirs("data/raw", exist_ok=True)
urllib.request.urlretrieve(URL, "data/raw/processed.cleveland.data")
print("Downloaded. Note: no header, '?' = missing, target 0=no disease, 1-4=disease")