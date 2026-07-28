from pydantic import BaseModel
from typing import Dict, List

class DiseaseAssessment(BaseModel):
    probability: float
    threshold: float
    is_positive: bool

class PredictionResponse(BaseModel):
    predictions: Dict[str, DiseaseAssessment]
    detected_findings: List[str]
    heatmap_overlay: str
    model_version: str = "1.0.0-INT8"

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool