# service_training/app/preprocessing/cleaner.py
import pandas as pd

from .common.logger_config import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """
    A class to perform common data cleaning operations on a pandas DataFrame.

    This cleaner can handle missing values (by filling or dropping) and
    remove duplicate rows. The cleaning behavior can be configured during
    initialization.
    """

    def __init__(
        self,
        missing_policy: str = "fill",
        fill_value: str = "Uncategorized",
    ):
        """
        Initializes the DataCleaner with a specified cleaning strategy.

        Args:
            missing_policy (str, optional): The strategy for handling missing values.
                                            Supported policies:
                                            - "fill": Replace missing values with `fill_value`.
                                            - "drop": Remove rows containing any missing values.
                                            Defaults to "fill".
            fill_value (str, optional): The value to use when `missing_policy` is "fill".
                                        Defaults to "Uncategorized".
        """
        self.fill_value = fill_value
        self.missing_policy = missing_policy
        logger.info(
            f"DataCleaner initialized with missing_policy: '{self.missing_policy}' "
            f"and fill_value: '{self.fill_value}'"
        )

    def _remove_duplicates(self, row_data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes complete duplicate rows from the DataFrame.

        Args:
            row_data (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: A new DataFrame with duplicate rows removed.
        """
        initial_rows = row_data.shape[0]
        logger.debug(f"Checking for duplicates in {initial_rows} rows.")

        df = row_data.drop_duplicates()

        removed_count = initial_rows - df.shape[0]
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate row(s).")
        else:
            logger.debug("No duplicate rows found.")

        return df

    def _fill_missing_values(self, row_data: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values (NaN) in the DataFrame based on the chosen policy.

        Args:
            row_data (pd.DataFrame): The DataFrame to process.

        Returns:
            pd.DataFrame: A DataFrame with missing values handled.

        Raises:
            ValueError: If an unsupported `missing_policy` is used.
        """
        if self.missing_policy == "fill":
            logger.debug(
                f"Applying 'fill' policy for missing values with value: '{self.fill_value}'"
            )
            df = row_data.fillna(value=self.fill_value)
        elif self.missing_policy == "drop":
            initial_rows = row_data.shape[0]
            logger.debug("Applying 'drop' policy for missing values.")
            df = row_data.dropna()
            removed_count = initial_rows - df.shape[0]
            if removed_count > 0:
                logger.info(f"Dropped {removed_count} row(s) with missing values.")
            else:
                logger.debug("No rows with missing values found to drop.")
        else:
            logger.error(f"Invalid missing policy specified: '{self.missing_policy}'")
            raise ValueError(f"Invalid missing policy: {self.missing_policy}")

        return df

    def clean_data(self, row_data: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the full data cleaning pipeline on a DataFrame.

        This method first removes duplicates and then handles missing values
        according to the policies defined during initialization.

        Args:
            row_data (pd.DataFrame): The raw DataFrame to be cleaned.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        logger.info(
            f"Starting data cleaning process for a DataFrame with {row_data.shape[0]} rows."
        )

        # Step 1: Remove duplicates
        df = self._remove_duplicates(row_data)

        # Step 2: Handle missing values
        df = self._fill_missing_values(df)

        logger.info(f"Data cleaning complete. Output DataFrame has {df.shape[0]} rows.")
        return df
