# Assignment Walkthrough Video Script

**Target length:** 4–5 minutes  
**Presenter:** Hussain Badri  
**Project:** MLOps Assignment 1: Heart Disease Prediction

Use the bracketed text as a screen cue; the quoted paragraphs are the narration. Speak naturally and keep each screen visible long enough to read.

## 0:00–0:25 | Introduction

**Show:** GitHub repository home page.

> Hello, my name is Hussain Badri, student ID 2025AF05063. This is my MLOps Assignment 1 project: an end-to-end heart-disease prediction pipeline using the UCI Cleveland dataset. I’ll briefly show the data and modeling workflow, experiment tracking, CI/CD, API deployment, and monitoring.

## 0:25–0:55 | Architecture and Data

**Show:** `screenshots/architecture_diagram.png`, then `data/download.py` and the EDA notebook.

> The pipeline starts with the UCI Cleveland data, which is downloaded into the project’s raw-data folder and explored in a Jupyter notebook. The source dataset has 303 records and 13 input features. Six rows with missing values are removed, leaving 297 records. The original target is converted to binary: zero means no disease, and values one through four are grouped as disease.

## 0:55–1:45 | Modeling and MLflow

**Show:** `src/preprocess.py`, `src/train.py`, then the MLflow experiment’s Training runs page.

> Preprocessing standardizes five numeric features and passes the integer-coded categorical features through unchanged. Training uses a stratified 80/20 holdout split with a fixed random seed, and compares Logistic Regression with Random Forest. These figures are from that holdout split, not five-fold cross-validation. MLflow records the metrics and model artifacts.
>
> Logistic Regression reached an ROC-AUC of about 0.949, while Random Forest reached about 0.939. Random Forest had higher accuracy on this split, but the script selects the model by ROC-AUC, so Logistic Regression was saved as `best_model.joblib` for the API.

**Show:** `screenshots/mlflow.png`; briefly point to the successful run entries. If metrics are not visible in the table, show the run details or the comparison table in `REPORT.md`.

## 1:45–2:20 | CI/CD and Container

**Show:** `.github/workflows/ci-cd.yml`, followed by the successful GitHub Actions run and GHCR package page.

> GitHub Actions runs linting and unit tests first. If those pass, a second job downloads the data, trains the model, builds the Docker image, and publishes it to GitHub Container Registry with `latest` and `v3` tags. This workflow publishes the image; the Minikube deployment is applied separately.

## 2:20–3:05 | API Prediction

**Show:** `src/api.py`, then the PowerShell request and response in `screenshots/curl_output.png`.

> The FastAPI service loads the selected model and exposes `/predict`, `/health`, and `/metrics`. For the sample patient shown here, the API returns a model-estimated risk of about 0.034 and a binary prediction of zero. This is only a software demonstration and should not be used for medical decisions.

## 3:05–3:45 | Kubernetes Deployment

**Show:** `k8s/deployment.yaml`, `k8s/service.yaml`, then `screenshots/kubectl get pods.png`.

> The container is deployed to a local Minikube Kubernetes cluster. The deployment uses the GHCR `v3` image, a readiness probe on `/health`, and a ClusterIP service on port 8000. The live deployment has four ready API replicas, alongside the monitoring pods. The checked-in deployment manifest requests two replicas, so I scaled the live deployment to four for this demonstration.

## 3:45–4:25 | Monitoring and Close

**Show:** `screenshots/promethus.png`, then `screenshots/grafana.png`.

> Prometheus scrapes the API’s `/metrics` endpoint every 15 seconds, including the request counter and prediction-latency histogram. The target is shown as up, and Grafana visualizes the API’s request and latency metrics. This completes the path from source data to an observable deployed service. Thank you.

## Recording Checklist

- Keep the video between 3 and 5 minutes; this script is paced for about 4 minutes.
- Use the screenshots already in `screenshots/` to avoid waiting for services to load during recording.
- If showing live dashboards, verify MLflow is open on port 5000 and Minikube, Grafana, and Prometheus are running beforehand.
- Show the current successful GitHub Actions run, not the earlier permission-denied attempt.
- Avoid showing access tokens, registry credentials, or secret values on screen.
- End with the repository URL visible: https://github.com/2025af05063-a11y/MlopsAssignment1
