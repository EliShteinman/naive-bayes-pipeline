# backend/main.py
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from nb_classifier.application_manager import (
    extract_expected_features,
    prepare_model_pipeline,
)
from nb_classifier.logger_config import get_logger

# Pydantic's BaseModel is no longer needed for the predict endpoint
# from pydantic import BaseModel

logger = get_logger(__name__)

# --- Global objects to be managed by lifespan ---
ml_models = {}


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. Code before 'yield' runs on startup,
    and code after 'yield' runs on shutdown.
    """
    # Startup: Load the model and prepare resources
    logger.info("Server startup: Loading model and preparing application...")
    try:
        classifier, trained_model = prepare_model_pipeline()
        expected_features = extract_expected_features(trained_model)

        ml_models["classifier"] = classifier
        ml_models["expected_features"] = expected_features

        logger.info("Model loaded and application is ready.")
    except (FileNotFoundError, RuntimeError, Exception) as e:
        logger.critical(f"A critical error occurred during startup: {e}", exc_info=True)
        ml_models["error"] = "Model could not be loaded. Check server logs."

    yield

    # Shutdown: Clean up the resources
    logger.info("Server shutdown: Clearing ML models and resources.")
    ml_models.clear()


# --- FastAPI App Initialization with Lifespan ---
app = FastAPI(
    title="Mushroom Classifier API",
    description="An API to predict if a mushroom is poisonous or edible based on its features.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- API Endpoints ---
@app.get("/expected-features", response_model=Dict[str, Any])
def get_expected_features():
    """
    Returns the feature schema of the model.
    """
    if ml_models.get("error"):
        raise HTTPException(status_code=503, detail=ml_models["error"])
    return ml_models["expected_features"]


@app.get("/predict", response_model=Dict[str, Any])
def predict(request: Request):
    """
    Predicts the class of a mushroom based on its features passed as query parameters.

    Example Usage:
    /predict?cap-shape=x&cap-surface=s&cap-color=n&bruises=t&odor=p... (and so on for all features)
    """
    if ml_models.get("error"):
        raise HTTPException(status_code=503, detail=ml_models.get("error"))

    # Dynamically get all features from the query parameters
    features = dict(request.query_params)
    logger.info(f"Received GET prediction request for features: {features}")

    # --- Manual Validation ---
    # Check for missing features
    expected_feature_names = ml_models["expected_features"].keys()
    missing_features = set(expected_feature_names) - set(features.keys())
    if missing_features:
        detail = (
            f"Missing required query parameters: {', '.join(sorted(missing_features))}"
        )
        logger.warning(f"Prediction failed due to bad input: {detail}")
        raise HTTPException(status_code=400, detail=detail)

    # Check for extra features that the model doesn't know
    extra_features = set(features.keys()) - set(expected_feature_names)
    if extra_features:
        detail = f"Unknown parameters provided: {', '.join(sorted(extra_features))}"
        logger.warning(f"Prediction failed due to bad input: {detail}")
        raise HTTPException(status_code=400, detail=detail)

    # --- Prediction Logic ---
    classifier = ml_models["classifier"]
    try:
        prediction = classifier.predict(features)
        logger.info(f"Prediction successful: {prediction}")
        return prediction
    except ValueError as e:
        # This will catch errors from the classifier, e.g., an unknown feature value
        logger.warning(f"Prediction failed due to bad value in input: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during prediction: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )


if __name__ == "__main__":
    logger.info("Running application in development mode.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
