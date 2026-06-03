import joblib
import pandas as pd
from fastapi import FastAPI

from src.api.pydantic_models import PredictionRequest, PredictionResponse


app = FastAPI(
    title="Credit Risk Scoring API",
    description="API for predicting customer credit risk using RFM features",
    version="1.0.0",
)

model = joblib.load("models/best_model.pkl")


@app.get("/")
def root():
    return {"message": "Credit Risk Scoring API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    input_data = pd.DataFrame(
        [
            {
                "recency": request.recency,
                "frequency": request.frequency,
                "monetary": request.monetary,
            }
        ]
    )

    risk_probability = model.predict_proba(input_data)[0][1]

    credit_score = int(850 - (risk_probability * 550))

    risk_label = "High Risk" if risk_probability >= 0.5 else "Low Risk"

    return PredictionResponse(
        risk_probability=round(float(risk_probability), 4),
        credit_score=credit_score,
        risk_label=risk_label,
    )