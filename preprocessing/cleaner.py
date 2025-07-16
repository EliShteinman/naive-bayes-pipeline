# preprocessing/cleaner.py
import pandas as pd
from config.logger_config import get_logger

logger = get_logger(__name__)


class DataCleaner:
    def __init__(
        self,
        missing_policy: str = "fill",
        fill_value: str = "Uncategorized",
    ):
        self.fill_value = fill_value
        self.missing_policy = missing_policy
        logger.debug(
            f"DataCleaner initialized with missing_policy: {missing_policy} "
            f"and fill_value: {fill_value}"
        )

    def _remove_duplicates(self, row_data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.
        Args:
            row_data: pd.DataFrame
        Returns: pd.DataFrame with duplicates removed
        """
        logger.debug(f"Removing duplicates from {row_data.shape[0]} rows")
        return row_data.drop_duplicates()

    def _fill_missing_values(self, row_data: pd.DataFrame) -> pd.DataFrame:
        if self.missing_policy == "fill":
            logger.debug(f"Filling missing values with {self.fill_value}")
            df = row_data.fillna(value=self.fill_value)
        elif self.missing_policy == "drop":
            logger.debug("Dropping missing values")
            df = row_data.dropna()
        else:
            logger.error(f"Invalid missing policy: {self.missing_policy}")
            raise ValueError(f"Invalid missing policy: {self.missing_policy}")
        return df

    def clean_data(self, row_data: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"Cleaning data with {row_data.shape[0]} rows")
        original_rows = row_data.shape[0]

        df = self._remove_duplicates(row_data)
        logger.debug(
            f"Removed duplicates: {original_rows - df.shape[0]} rows removed, "
            f"{df.shape[0]} remaining"
        )

        df = self._fill_missing_values(df)
        if self.missing_policy == "fill":
            logger.debug(f"Filled missing values with '{self.fill_value}'")
        elif self.missing_policy == "drop":
            logger.debug("Dropped rows with missing values")

        return df
