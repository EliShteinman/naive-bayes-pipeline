# model_trainer/main.py
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException

from app.application_manager import prepare_model_pipeline
from app.model_artifact import IModelArtifact, NaiveBayesDictArtifact
from app.logger_config import get_logger

logger = get_logger(__name__)

# --- Global object to hold the trained model ---
model_store = {}


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    On startup, run the training pipeline and store the resulting artifact.
    """
    logger.info("Trainer service startup: Starting model training pipeline...")
    try:
        # 1. Run the pipeline to get the trained artifact object
        trained_model_artifact: IModelArtifact = prepare_model_pipeline(
            file_path="data/mushroom_decoded.csv",
            target_col="poisonous",
            pos_label="p"
        )

        # 2. Store the artifact in the global store
        model_store["artifact"] = trained_model_artifact
        logger.info("Model training pipeline completed successfully. Artifact is ready.")

    except (FileNotFoundError, RuntimeError) as e:
        logger.critical(f"A critical error occurred during training: {e}", exc_info=True)
        model_store["error"] = f"Model training failed: {e}"
    except Exception as e:
        logger.critical(f"An unexpected error occurred during training: {e}", exc_info=True)
        model_store["error"] = "An unexpected internal error occurred."

    yield

    # Shutdown: Clean up resources
    logger.info("Trainer service shutdown.")
    model_store.clear()


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Model Trainer Service",
    description="An internal service that trains a model and exposes it for serving.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- API Endpoints ---
@app.get("/latest-model", response_model=Dict[str, Any])
def get_latest_model():
    """
    Returns the data dictionary of the latest successfully trained model.
    """
    if "error" in model_store:
        raise HTTPException(status_code=503, detail=model_store["error"])

    artifact = model_store.get("artifact")
    if artifact is None:
        raise HTTPException(status_code=404, detail="Model artifact not available yet.")

    # Convert the artifact object to a serializable dictionary
    if isinstance(artifact, NaiveBayesDictArtifact):
        return artifact.to_dict()
    else:
        # Handle cases where the artifact might be of a different, unsupported type
        error_msg = f"Artifact type {type(artifact).__name__} is not supported for serialization."
        logger.error(error_msg)
        raise HTTPException(status_code=501, detail=error_msg)


# To run this service directly for development
if __name__ == "__main__":
    logger.info("Running Model Trainer Service in development mode.")
    # Port 8001 to avoid conflict with the model_server on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)