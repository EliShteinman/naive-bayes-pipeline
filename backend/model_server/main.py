from contextlib import asynccontextmanager
from typing import Any, Dict, List, Type

import requests
import uvicorn
from app import get_logger, model_artifact
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from . import config

logger = get_logger(__name__)


# --- Global objects to be managed by lifespan ---
ml_models = {}


def load_model_from_url(url: str) -> None:
    logger.info(f"Loading model from {url}")
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch model. Status code: {response.status_code}"
        )

    artifact = model_artifact.NaiveBayesDictArtifact(response.json())
    ml_models["classifier"] = artifact
    ml_models["expected_features"] = artifact.get_schema()
    ml_models["model_url"] = url
    ml_models["error"] = None


def create_dynamic_model_from_schema(
    model_name: str, schema: Dict[str, List[str]]
) -> Type[BaseModel]:
    """
    Creates a Pydantic model dynamically from a given schema dictionary.
    """
    fields = {}
    for feature_name in schema.keys():
        clean_name = feature_name.replace("-", "_")
        fields[clean_name] = (str, Field(..., alias=feature_name))

    DynamicModel = create_model(model_name, **fields)
    return DynamicModel


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server service startup...")
    if not config.MODEL_URL:
        ml_models["error"] = "MODEL_URL is not configured. Cannot load model."
        logger.critical(ml_models["error"])
        yield
        return

    try:
        load_model_from_url(config.MODEL_URL)
        if "classifier" in ml_models:
            schema = ml_models["expected_features"]
            DynamicFeatureModel = create_dynamic_model_from_schema(
                "DynamicInputModel", schema
            )
            ml_models["pydantic_model"] = DynamicFeatureModel
            logger.info("Successfully created a dynamic Pydantic model.")
    except Exception as e:
        logger.critical(f"Model load failed: {e}", exc_info=True)
        ml_models["error"] = "Model failed to load during startup."

    yield

    logger.info("Server shutdown: Clearing ML models and resources.")
    ml_models.clear()


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
def predict(features: BaseModel = Depends(lambda: ml_models.get("pydantic_model"))):
    """
    Predicts the class of a mushroom based on its features passed as query parameters.

    Example Usage:
    /predict?cap-shape=x&cap-surface=s&cap-color=n&bruises=t&odor=p... (and so on for all features)
    """
    if ml_models.get("error"):
        raise HTTPException(status_code=503, detail=ml_models.get("error"))

    if not features:
        raise HTTPException(
            status_code=503, detail="Model not ready. Please try again."
        )

    feature_dict = features.dict(by_alias=True)
    logger.info(f"Received GET prediction request for features: {feature_dict}")

    classifier = ml_models["classifier"]
    try:
        prediction = classifier.predict(feature_dict)
        logger.info(f"Prediction successful: {prediction}")
        return prediction
    except ValueError as e:
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

        # After reloading, we must also recreate the dynamic Pydantic model
        if "classifier" in ml_models:
            schema = ml_models["expected_features"]
            DynamicFeatureModel = create_dynamic_model_from_schema(
                model_name="DynamicInputModel", schema=schema
            )
            ml_models["pydantic_model"] = DynamicFeatureModel
            logger.info("Re-created dynamic Pydantic model after reload.")

        return {"status": "success", "message": "Model reloaded successfully."}
    except Exception as e:
        logger.error(f"Model reload failed: {e}", exc_info=True)
        ml_models["error"] = "Model reload failed."
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Running application in development mode.")
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
