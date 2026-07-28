from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import PredictionResponse, HealthResponse
from src.inference.predictor import CXRPredictor

app = FastAPI(
    title="ExplainX-CXR API",
    description="Multi-label Chest X-Ray pathology diagnosis endpoint",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate global predictor instance
try:
    predictor = CXRPredictor()
except Exception as e:
    predictor = None
    print(f"Warning: Model failed to load at startup: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy" if predictor is not None else "degraded",
        "model_loaded": predictor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_xray(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Model is not initialized.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        
    try:
        image_bytes = await file.read()
        results = predictor.predict(image_bytes)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")