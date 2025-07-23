# backend/nb_classifier/data_cleaner.py
import pandas as pd

from .logger_config import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """
    Handles cleaning operations on a pandas DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the DataCleaner with the DataFrame to be cleaned.

        Args:
            data (pd.DataFrame): The raw DataFrame.
        """
        self.data = data

    def clean(self) -> pd.DataFrame:
        """
        Applies a series of cleaning steps to the DataFrame.

        The current implementation performs two main actions:
        1. Drops the 'stalk-root' column if it exists.
        2. Removes any columns that have only one unique value (i.e., no variance).

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        logger.info("Starting data cleaning process...")

        # Drop columns that are known to be problematic or have been decided to be excluded.
        # 'errors="ignore"' prevents an error if the column doesn't exist.
        cleaned_data = self.data.drop(columns=["stalk-root"], errors="ignore")

        # Remove columns with only one unique value, as they provide no information for the model.
        # The 'lambda df: df.nunique() > 1' is applied to the columns axis.
        original_cols = set(cleaned_data.columns)
        cleaned_data = cleaned_data.loc[:, lambda df: df.nunique() > 1]
        cleaned_cols = set(cleaned_data.columns)

        removed_cols = original_cols - cleaned_cols
        if removed_cols:
            logger.info(f"Removed constant columns: {', '.join(removed_cols)}")

        logger.info("Data cleaning finished.")
        return cleaned_data
