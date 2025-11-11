"""app.py
FastAPI application for hospital readmission prediction.
Loads trained model and preprocessing pipeline for secure inference.
"""
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY", "Laban047")  # Example security key

# Initialize FastAPI app
app = FastAPI(title="Hospital Readmission Predictor", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security dependency
def verify_api_key(request: Request):
    key = request.headers.get("x-api-key")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")

# Load model and pipeline
MODEL_PATH = 'models/readmission_model.joblib'
PIPELINE_PATH = 'models/preprocessing_pipeline.joblib'

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Model load failed: {e}")

# Input schema
class PatientData(BaseModel):
    age: int
    gender: str
    length_of_stay: float
    prior_admissions: int
    creatinine: float | None = None
    hemoglobin: float | None = None
    glucose: float | None = None
    medications: str | None = None
    icd_codes: str | None = None
    follow_up_date: str | None = None

@app.get("/")
def root():
    return {"message": "Hospital Readmission Prediction API is running."}

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    df = pd.DataFrame([data.dict()])

    try:
        # Predict probability of readmission
        proba = model.predict_proba(df)[:, 1][0]
        prediction = int(proba >= 0.5)
        return {
            "readmission_risk": prediction,
            "probability": round(float(proba), 3),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
