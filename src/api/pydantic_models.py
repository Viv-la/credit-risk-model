from pydantic import BaseModel


class PredictionRequest(BaseModel):
    recency: float
    frequency: float
    monetary: float


class PredictionResponse(BaseModel):
    risk_probability: float
    credit_score: int
    risk_label: str