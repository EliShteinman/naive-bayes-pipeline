# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

from nb_classifier.data_handler import DataHandler
from nb_classifier.naive_bayes_model_builder import NaiveBayesModelBuilder
from nb_classifier.classifier import ClassifierService
from nb_classifier.logger_config import get_logger

# --- קבועים ---
FILE_PATH = '/Users/lyhwstynmn/פרוייקטים/python/naive-bayes-pipeline/data/mushroom_decoded.csv'
TARGET_COL = 'poisonous'

logger = get_logger(__name__)
app = FastAPI()


# --- שלב 1: בניית המודל פעם אחת בלבד ---
logger.info("Preparing and training model...")
data_handler = DataHandler(data_path=FILE_PATH)
train_data, _ = data_handler.get_split_data_as_dicts(target_col=TARGET_COL)

model_builder = NaiveBayesModelBuilder(alpha=1.0)
trained_model = model_builder.build_model(train_data, target_col=TARGET_COL)
classifier = ClassifierService(model_artifact=trained_model)
logger.info("Model ready.")


# --- שלב 2: חילוץ תכונות וערכים ---
def extract_expected_features(model: dict) -> dict:
    example_class = next(iter(model.values()))
    return {
        feature: sorted(values.keys())
        for feature, values in example_class.items()
        if feature != "__prior__"
    }

expected_features = extract_expected_features(trained_model)


# --- שלב 3: סכמת קלט לדגימה ---
class SampleInput(BaseModel):
    features: Dict[str, Any]


# --- ראוטים ---
@app.get("/expected-features")
def get_expected_features():
    return expected_features


@app.post("/predict")
def predict(input: SampleInput):
    try:
        prediction = classifier.predict(input.features)
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- שלב 4: הפעלת השרת ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)