# service_data_loader/app/main_api.py

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .common.logger_config import get_logger, setup_logger
from .data_loader import Loader

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
    title="Data Loader Service",
    description="An API to load data from various file formats and return it as JSON.",
    version="1.0.0",
    lifespan=lifespan,
)


class DataLoaderInput(BaseModel):
    """Defines the expected input for the /load endpoint."""

    source: str
    input_type: str = "csv"
    encoding: str = "utf-8"


class DataLoaderOutput(BaseModel):
    """
    Defines the output structure. The data is a list of dictionaries,
    where each dictionary is a row.
    """

    data: List[Dict[str, Any]]


@app.post("/load", response_model=DataLoaderOutput)
def load(input_data: DataLoaderInput) -> Dict:
    """
    Loads data from a specified source file and returns it as a list of records.

    The 'source' path should be the path *inside the container* (e.g., /app/data/dataset.csv).
    """
    start_time = time.time()
    logger.info(f"Received request to load data from: {input_data.source}")

    try:
        loader = Loader(
            source=input_data.source,
            input_type=input_data.input_type,
            encoding=input_data.encoding,
        )
        data = loader.load()

        # Convert DataFrame to a list of dictionaries, which is JSON-friendly
        data_as_list_of_dicts = data.to_dict(orient="records")

        duration = time.time() - start_time
        logger.info(f"Successfully loaded {len(data)} rows in {duration:.2f} seconds.")

        return {"data": data_as_list_of_dicts}

    except FileNotFoundError:
        logger.error(f"File not found at path: {input_data.source}")
        raise HTTPException(
            status_code=404,
            detail=f"File not found at the specified source path: {input_data.source}",
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during data loading: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@app.get("/health", status_code=200)
def health_check() -> Dict[str, str]:
    """A simple endpoint to check if the service is running."""
    logger.debug("Health check successful.")
    return {"status": "ok"}
