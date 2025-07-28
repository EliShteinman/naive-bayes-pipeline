from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from app.model_artifact import IModelArtifact, NaiveBayesDictArtifact
from app.logger_config import get_logger
from application_manager import prepare_model_pipeline
from fastapi import FastAPI, HTTPException

import config

logger = get_logger(__name__)

# --- Global state ---
model_store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Trainer service startup: Starting model training pipeline...")
    if not all([config.FILE_PATH, config.TARGET_COL, config.POS_LABEL]):
        model_store["error"] = (
            "Critical configuration (DATA_FILE_PATH, TARGET_COL, POS_LABEL) is missing."
        )
        logger.critical(model_store["error"])
        yield
        return

    try:
        trained_model_artifact: IModelArtifact = prepare_model_pipeline(
            file_path=config.FILE_PATH,
            target_col=config.TARGET_COL,
            pos_label=config.POS_LABEL,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            validate_test_set=config.VALIDATE_TEST_SET,
            min_accuracy=config.MIN_ACCURACY,
            columns_to_drop=config.COLUMNS_TO_DROP,
            alpha=config.ALPHA,
        )
        model_store["artifact"] = trained_model_artifact
        logger.info("Model training pipeline completed successfully.")
    except Exception as e:
        logger.critical(
            f"A critical error occurred during training: {e}", exc_info=True
        )
        model_store["error"] = f"Model training failed: {e}"

    yield

    logger.info("Trainer service shutdown.")
    model_store.clear()


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
