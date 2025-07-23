# backend/nb_classifier/data_cleaner.py
from typing import List, Optional

import pandas as pd

from .logger_config import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """
    Handles cleaning operations on a pandas DataFrame.
    The specific operations are configured during initialization.
    """

    def __init__(
        self, columns_to_drop: Optional[List[str]] = None, remove_constants: bool = True
    ):
        """
        Initializes the DataCleaner with a specific cleaning configuration.

        Args:
            columns_to_drop (Optional[List[str]]): A list of column names to drop.
                                                    Defaults to None.
            remove_constants (bool): If True, removes columns with only one unique value.
                                     Defaults to True.
        """
        self.columns_to_drop = columns_to_drop or []
        self.remove_constants = remove_constants
        logger.info(
            f"DataCleaner initialized. Drop columns: {self.columns_to_drop}, Remove constants: {self.remove_constants}"
        )

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the configured cleaning steps to the DataFrame.

        Args:
            data (pd.DataFrame): The raw DataFrame to be cleaned.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        logger.info("Starting data cleaning process...")
        cleaned_data = data.copy()

        # Step 1: Drop specified columns
        if self.columns_to_drop:
            logger.debug(f"Dropping specified columns: {self.columns_to_drop}")
            cleaned_data = cleaned_data.drop(
                columns=self.columns_to_drop, errors="ignore"
            )

        # Step 2: Remove constant columns
        if self.remove_constants:
            logger.debug("Removing constant columns...")
            cols_before = set(cleaned_data.columns)
            cleaned_data = cleaned_data.loc[:, cleaned_data.nunique() > 1]
            cols_after = set(cleaned_data.columns)

            removed_cols = cols_before - cols_after
            if removed_cols:
                logger.info(f"Removed constant columns: {sorted(list(removed_cols))}")

        logger.info("Data cleaning finished.")
        return cleaned_data
