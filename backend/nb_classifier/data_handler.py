# backend/nb_classifier/data_handler.py
import pandas as pd

from .logger_config import get_logger

logger = get_logger(__name__)


class DataHandler:
    """
    Handles loading data from a specified file path.
    """

    def __init__(self, data_path: str):
        """
        Initializes the DataHandler with the path to the data file.

        Args:
            data_path (str): The path to the CSV file to be loaded.
        """
        if not data_path:
            raise ValueError("Data path cannot be empty.")
        self.data_path = data_path
        logger.info(f"DataHandler initialized for path: {self.data_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded data.

        Raises:
            FileNotFoundError: If the file is not found at the specified path.
            Exception: For other errors during file loading.
        """
        logger.info(f"Attempting to load data from {self.data_path}")
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found at path: {self.data_path}")
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while loading data: {e}", exc_info=True
            )
            raise
