# service_preprocessor/app/main_api.py
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, cast

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .cleaner import DataCleaner
from .common.logger_config import get_logger, setup_logger
from .splitter import Splitter

logger = get_logger(__name__)


# --- Lifespan and App Initialization ---


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages the application's lifespan for startup and shutdown events."""
    setup_logger()
    logger.info("Logger has been configured. Preprocessor service is starting.")
    yield
    logger.info("Preprocessor service is shutting down.")


app = FastAPI(
    title="Preprocessor Service",
    description="An API to clean and split data for a machine learning pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)


# --- Pydantic Models for API I/O ---


class CleanDataInput(BaseModel):
    data: List[Dict[str, Any]]
    missing_policy: str = "fill"
    fill_value: str = "Uncategorized"


class CleanDataOutput(BaseModel):
    cleaned_data: List[Dict[str, Any]]


class SplitDataInput(BaseModel):
    data: List[Dict[str, Any]]
    target_col: str
    test_size: float = 0.3
    random_state: int = 42
    stratify: bool = True


class SplitDataOutput(BaseModel):
    train_data: List[Dict[str, Any]]
    test_data: List[Dict[str, Any]]


# --- Helper Function ---


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


# --- API Endpoints ---


@app.post("/clean_data", response_model=CleanDataOutput)
async def clean_data(input_data: CleanDataInput) -> CleanDataOutput:
    """Cleans the provided dataset according to the specified policy."""
    logger.info(
        f"Received request to clean data. Policy: '{input_data.missing_policy}', "
        f"Fill value: '{input_data.fill_value}'."
    )
    try:
        # Step 1: Convert input to DataFrame
        df = _convert_to_dataframe(input_data.data)

        # Step 2: Initialize and run the cleaner
        cleaner = DataCleaner(
            missing_policy=input_data.missing_policy,
            fill_value=input_data.fill_value,
        )
        cleaned_df = cleaner.clean_data(df)

        # Step 3: Convert result back to JSON-friendly format
        records = cast(List[Dict[str, Any]], cleaned_df.to_dict(orient="records"))
        logger.info(
            f"Successfully cleaned data. Input rows: {len(df)}, Output rows: {len(records)}."
        )

        return CleanDataOutput(cleaned_data=records)

    except HTTPException:
        # Re-raise HTTPException to preserve status code and detail
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during data cleaning: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        )


@app.post("/split_data", response_model=SplitDataOutput)
async def split_data(input_data: SplitDataInput) -> SplitDataOutput:
    """Splits the provided dataset into training and testing sets."""
    logger.info(
        f"Received request to split data for target: '{input_data.target_col}'."
    )
    try:
        # Step 1: Convert input to DataFrame
        df = _convert_to_dataframe(input_data.data)

        # Step 2: Initialize and run the splitter
        splitter = Splitter(
            test_size=input_data.test_size,
            random_state=input_data.random_state,
            stratify=input_data.stratify,
        )
        train_df, test_df = splitter.split(df, input_data.target_col)

        # Step 3: Convert results back to JSON-friendly format
        train_records = cast(List[Dict[str, Any]], train_df.to_dict(orient="records"))
        test_records = cast(List[Dict[str, Any]], test_df.to_dict(orient="records"))

        logger.info(
            f"Successfully split data. Train set: {len(train_records)} rows, "
            f"Test set: {len(test_records)} rows."
        )

        return SplitDataOutput(train_data=train_records, test_data=test_records)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during data splitting: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"An internal server error occurred: {e}"
        )


@app.get("/health", status_code=200)
def health_check() -> Dict[str, str]:
    """A simple endpoint to check if the service is running."""
    logger.debug("Health check successful.")
    return {"status": "ok"}
