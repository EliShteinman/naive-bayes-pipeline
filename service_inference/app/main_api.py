# service_inference/app/main_api.py

import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import FeatureValues, Model, Predictor, get_logger, setup_logger

model: Model | None = None
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the application's lifespan. Code before 'yield' runs on startup,
    and code after 'yield' runs on shutdown.
    """
    setup_logger()
    logger.info("Logger has been configured.")

    global model
    artifact_path = "shared_artifacts/model.json"
    try:
        logger.info(f"Loading model from: {artifact_path}")
        with open(artifact_path, "r") as f:
            model = json.load(f)
        logger.info("Model loaded successfully.")
    except FileNotFoundError:
        logger.error(
            f"FATAL: Model not found at {artifact_path}. API cannot serve predictions."
        )
    except json.JSONDecodeError:
        logger.error(
            f"FATAL: Could not decode JSON from {artifact_path}. File might be corrupted."
        )

    yield
    logger.info("Application is shutting down.")


app = FastAPI(
    title="Naive Bayes Classifier API",
    description="An API to predict outcomes based on a pre-trained Naive Bayes model.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionInput(BaseModel):
    features: Dict[str, Any]


class PredictionOutput(BaseModel):
    prediction: str


class SchemaOutput(BaseModel):
    features: FeatureValues
    possible_outcomes: List[str]


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput) -> Dict:
    start_time = time.time()
    logger.info("Received a new prediction request.")
    logger.debug(f"Request payload: {input_data.dict()}")

    if model is None:
        logger.error("Prediction request failed because model is not loaded.")
        raise HTTPException(
            status_code=503,  # Service Unavailable
            detail="Model is not loaded. Please try again later.",
        )

    instance = pd.Series(input_data.features)

    try:
        prediction_result = Predictor.predict_instance(instance, model)

        duration = (time.time() - start_time) * 1000  # in milliseconds
        logger.info(
            f"Prediction successful. Result: '{prediction_result}'. Duration: {duration:.2f}ms"
        )

        return {"prediction": prediction_result}
    except KeyError as e:
        logger.warning(
            f"Prediction failed due to an unseen feature/value from client. Error: {e}"
        )
        raise HTTPException(
            status_code=400,  # Bad Request
            detail=f"Invalid input. An unseen feature or value was provided: {e}",
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during prediction: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,  # Internal Server Error
            detail="An internal error occurred during prediction.",
        )


@app.get("/schema", response_model=SchemaOutput)
def get_model_schema() -> Dict:
    logger.info("Received a request for model schema.")
    if model is None:
        logger.warning("Schema request failed because model is not loaded.")
        raise HTTPException(
            status_code=503, detail="Model is not loaded. Cannot provide schema."
        )

    feature_schema = model["feature_values"]
    possible_outcomes = list(model["target_priors"].keys())

    logger.debug(
        f"Returning schema with {len(feature_schema)} features and {len(possible_outcomes)} outcomes."
    )
    return {"features": feature_schema, "possible_outcomes": possible_outcomes}


@app.get("/health", status_code=200)
def health_check() -> Dict[str, str]:
    logger.debug("Health check successful.")
    return {"status": "ok"}
