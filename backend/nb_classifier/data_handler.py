# nb_classifier/data_handler.py
import pandas as pd
from .logger_config import get_logger

logger = get_logger(__name__)


class DataHandler:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully from {self.data_path}")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading data: {e}")
            raise

