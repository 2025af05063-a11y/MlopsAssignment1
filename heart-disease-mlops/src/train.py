import mlflow, mlflow.sklearn, joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

if __package__:
    from .preprocess import load_heart_data, build_preprocessor
else:
    from preprocess import load_heart_data, build_preprocessor

mlflow.set_experiment("heart-disease-classification")

df = load_heart_data()
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model, best_auc = None, 0

for name, clf in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        mlflow.log_param("model_type", name)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        
        # Log EDA plots as artifacts
        for plot in ["class_balance.png","correlation_heatmap.png","histograms.png"]:
            mlflow.log_artifact(f"notebooks/notebooks/plots/{plot}")
        
        if metrics["roc_auc"] > best_auc:
            best_auc, best_model = metrics["roc_auc"], pipeline

# Save best model with joblib for API
joblib.dump(best_model, "best_model.joblib")
print(f"Best ROC-AUC: {best_auc:.4f} — saved to best_model.joblib")