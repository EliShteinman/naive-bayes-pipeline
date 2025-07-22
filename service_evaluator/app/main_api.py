# service_evaluator/app/main_api.py

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .common import Model
from .common.logger_config import get_logger, setup_logger
from .evaluator import ModelEvaluator

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the application's lifespan. Code before 'yield' runs on startup,
    and code after 'yield' runs on shutdown.
    """
    setup_logger()
    logger.info("Logger has been configured. Data Loader service is starting.")
    yield
    logger.info("Data Loader service is shutting down.")


app = FastAPI(
    title="Model Evaluator Service",
    description="An API to evaluate a model's performance on a test dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


def _convert_to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Converts a list of dictionaries to a pandas DataFrame, with logging."""
    if not data:
        logger.warning("Input data for DataFrame conversion is empty.")
        raise HTTPException(status_code=400, detail="Input data cannot be empty.")

    logger.debug(f"Converting {len(data)} records into a pandas DataFrame.")
    try:
        df = pd.DataFrame(data)
        logger.debug(f"DataFrame created successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to convert input data to DataFrame: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data structure provided. Could not create DataFrame: {e}",
        )


class EvaluateInput(BaseModel):
    model: Model
    test_data: List[Dict[str, Any]]
    target_col: str


class EvaluateOutput(BaseModel):
    metrics: Dict[str, float]


@app.post("/evaluate", response_model=EvaluateOutput)
async def evaluate_model(input_data: EvaluateInput):
    logger.info(
        f"Received request to evaluate model: {input_data.model}"
        f" on test data with target column: {input_data.target_col}"
        f"  {input_data.test_data}"
        f" with {len(input_data.test_data)} rows"
    )
    try:
        df = _convert_to_dataframe(input_data.test_data)
        model = input_data.model
        target_col = input_data.target_col
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, df, target_col)
        return EvaluateOutput(metrics=metrics)

    except Exception as e:
        logger.error(f"Failed to convert input data to DataFrame: {e}", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data structure provided. Could not create DataFrame: {e}",
        )


@app.get("/health", status_code=200)
def health_check() -> Dict[str, str]:
    """A simple endpoint to check if the service is running."""
    logger.debug("Health check successful.")
    return {"status": "ok"}
