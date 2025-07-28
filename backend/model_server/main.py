# main.py
from contextlib import asynccontextmanager
from typing import Any, Dict
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from app import get_logger, model_artifact
import requests


logger = get_logger(__name__)

# --- Global objects to be managed by lifespan ---
ml_models = {}


def load_model_from_url(url: str) -> None:
    logger.info(f"Loading model from {url}")
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to fetch model. Status code: {response.status_code}")

    artifact = model_artifact.NaiveBayesDictArtifact(response.json())
    ml_models["classifier"] = artifact
    ml_models["expected_features"] = artifact.get_schema()
    ml_models["model_url"] = url
    ml_models["error"] = None

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        default_url = os.getenv("MODEL_URL")
        if default_url:
            load_model_from_url(default_url)
        else:
            ml_models["error"] = "No model loaded. URL is not configured."
            logger.warning("No default model URL provided at startup.")
    except Exception as e:
        logger.critical(f"Model load failed: {e}", exc_info=True)
        ml_models["error"] = "Model failed to load during startup."

    yield

    logger.info("Server shutdown: Clearing ML models and resources.")
    ml_models.clear()



# --- FastAPI App Initialization with Lifespan ---
app = FastAPI(
    title="Mushroom Classifier API (Server)",
    description="An API to predict if a mushroom is poisonous or edible.",
    version="2.0.0",
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


@app.post("/reload-model")
def reload_model(url: str = None):
    """
    Reloads the model from the last known URL or a new one.
    If `url` is passed — uses it; else uses the last loaded URL.
    """
    try:
        if url:
            load_model_from_url(url)
        elif ml_models.get("model_url"):
            load_model_from_url(ml_models["model_url"])
        else:
            raise HTTPException(status_code=400, detail="No model URL specified.")
        return {"status": "success", "message": "Model reloaded successfully."}
    except Exception as e:
        logger.error(f"Model reload failed: {e}", exc_info=True)
        ml_models["error"] = "Model reload failed."
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Running application in development mode.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
