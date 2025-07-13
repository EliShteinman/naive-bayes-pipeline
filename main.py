from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

from nb_classifier.application_manager import prepare_model_pipeline,extract_expected_features
from nb_classifier.logger_config import get_logger

logger = get_logger(__name__)
app = FastAPI()

# --- בניית המודל + בדיקת דיוק ---
classifier, trained_model = prepare_model_pipeline()

# --- חילוץ סכמת תכונות ---
expected_features = extract_expected_features(trained_model)

# --- סכמת קלט ---
class SampleInput(BaseModel):
    features: Dict[str, Any]

# --- ראוטים ---
@app.get("/expected-features")
def get_expected_features():
    return expected_features

@app.post("/predict")
def predict(input: SampleInput):
    try:
        return classifier.predict(input.features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)