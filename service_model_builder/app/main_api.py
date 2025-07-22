# service_model_builder/app/main_api.py
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .builder import NaiveBayesModelBuilder
from .common.logger_config import get_logger, setup_logger
from .common.typing_defs import Model

logger = get_logger(__name__)


# --- Lifespan and App Initialization ---


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages the application's lifespan for startup and shutdown events."""
    setup_logger()
    logger.info("Logger has been configured. Model Builder service is starting.")
    yield
    logger.info("Model Builder service is shutting down.")


app = FastAPI(
    title="Model Builder Service",
    description="An API to build a Naive Bayes model from training data.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic Models for API I/O ---


class BuildModelInput(BaseModel):
    train_data: List[Dict[str, Any]]
    target_col: str
    alpha: int = 1


class BuildModelOutput(BaseModel):
    model: Model


# --- Helper Function ---


def _convert_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts a list of dictionaries to a pandas DataFrame, with logging and error handling."""
    if not data:
        logger.warning("Input data for DataFrame conversion is empty.")
        raise HTTPException(
            status_code=400, detail="Input 'train_data' cannot be empty."
        )

    logger.debug(f"Converting {len(data)} records into a pandas DataFrame.")
    try:
        df = pd.DataFrame(data)
        logger.debug(
            f"DataFrame for training created successfully with shape: {df.shape}"
        )
        return df
    except Exception as e:
        logger.error(f"Failed to convert input data to DataFrame: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data structure provided. Could not create DataFrame: {e}",
        )


# --- API Endpoints ---


@app.post("/build_model", response_model=BuildModelOutput)
async def build_model(input_data: BuildModelInput) -> BuildModelOutput:
    """
    Builds a Naive Bayes model from the provided training data.
    """
    logger.info(
        f"Received request to build model for target '{input_data.target_col}' "
        f"with alpha={input_data.alpha}."
    )
    try:
        # Step 1: Convert input to DataFrame
        df = _convert_to_dataframe(input_data.train_data)

        # Step 2: Initialize and run the builder
        logger.debug("Initializing NaiveBayesModelBuilder.")
        builder = NaiveBayesModelBuilder(alpha=input_data.alpha)

        logger.info(f"Starting model build on {df.shape[0]} rows...")
        model = builder.build(df, input_data.target_col)

        logger.info("Model built successfully.")

        return BuildModelOutput(model=model)

    except HTTPException:
        # Re-raise HTTPException from the helper to preserve status code and detail
        raise
    except ValueError as e:
        # Catch specific errors from the builder, like "target column not found"
        logger.warning(f"Failed to build model due to invalid input: {e}")
        raise HTTPException(
            status_code=422, detail=f"Failed to build model: {e}"
        )  # 422 Unprocessable Entity
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while building the model: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        )


@app.get("/health", status_code=200)
def health_check() -> Dict[str, str]:
    """A simple endpoint to check if the service is running."""
    logger.debug("Health check successful.")
    return {"status": "ok"}
