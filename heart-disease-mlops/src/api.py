import joblib, time, logging
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="Heart Disease Predictor")
model = joblib.load("best_model.joblib")

# Prometheus metrics
REQUEST_COUNT = Counter("api_requests_total", "Total requests", ["endpoint"])
PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Latency")

class Patient(BaseModel):
    age: float; sex: int; cp: int; trestbps: float; chol: float
    fbs: int; restecg: int; thalach: float; exang: int
    oldpeak: float; slope: int; ca: float; thal: float

@app.post("/predict")
def predict(p: Patient):
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    start = time.time()
    features = [[p.age, p.sex, p.cp, p.trestbps, p.chol, p.fbs,
                 p.restecg, p.thalach, p.exang, p.oldpeak,
                 p.slope, p.ca, p.thal]]
    proba = model.predict_proba(features)[0][1]
    PREDICTION_LATENCY.observe(time.time() - start)
    return {"heart_disease_risk": float(proba),
            "prediction": int(proba > 0.5)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health(): return {"status": "ok"}