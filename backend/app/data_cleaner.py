# app/data_cleaner.py
from typing import Any, List, Literal, Optional

import pandas as pd

from app.logger_config import get_logger

logger = get_logger(__name__)

# Define types for strategies to make the code clearer
MissingValueStrategy = Literal["drop_row", "fill"]
FillMethod = Literal["mean", "median", "mode"]


class DataCleaner:
    """
    Handles cleaning operations on a pandas DataFrame.
    The specific operations are configured during initialization, including
    dropping columns, removing constants, and handling missing values.
    """

    def __init__(
        self,
        columns_to_drop: Optional[List[str]] = None,
        remove_constants: bool = True,
        missing_value_strategy: Optional[MissingValueStrategy] = None,
        fill_value: Optional[Any] = None,
    ):
        """
        Initializes the DataCleaner with a specific cleaning configuration.

        Args:
            columns_to_drop (Optional[List[str]]): A list of column names to drop.
            remove_constants (bool): If True, removes columns with only one unique value.
            missing_value_strategy (Optional[MissingValueStrategy]): How to handle missing values.
                - 'drop_row': to remove rows with any missing values.
                - 'fill': to impute them using the `fill_value`.
            fill_value (Optional[Any]): The value to use for imputation if strategy is 'fill'.
                - Can be a constant value (e.g., 0, "Unknown").
                - Can be a keyword ("mean", "median", "mode") to use pandas methods.
        """
        self.columns_to_drop = columns_to_drop or []
        self.remove_constants = remove_constants
        self.missing_value_strategy = missing_value_strategy
        self.fill_value = fill_value

        # A check to ensure that if the user wants to fill, they provide a value
        if self.missing_value_strategy == "fill" and self.fill_value is None:
            raise ValueError(
                "fill_value must be provided when missing_value_strategy is 'fill'"
            )

        logger.info(
            f"DataCleaner initialized. Configuration: "
            f"Drop Columns: {self.columns_to_drop}, "
            f"Remove Constants: {self.remove_constants}, "
            f"Missing Value Strategy: {self.missing_value_strategy}, "
            f"Fill Value: {self.fill_value}"
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

        # Step 3: Handle missing values based on the configured strategy
        if self.missing_value_strategy:
            logger.info(
                f"Handling missing values with strategy: '{self.missing_value_strategy}'"
            )

            if self.missing_value_strategy == "drop_row":
                rows_before = len(cleaned_data)
                cleaned_data.dropna(inplace=True)
                rows_after = len(cleaned_data)
                if rows_before > rows_after:
                    logger.warning(
                        f"Dropped {rows_before - rows_after} rows with missing values."
                    )

            elif self.missing_value_strategy == "fill":
                logger.info(
                    f"Attempting to fill missing values with: {self.fill_value}"
                )

                # If fill_value is a pandas method keyword (e.g., "mean")
                if isinstance(self.fill_value, str) and self.fill_value in (
                    "mean",
                    "median",
                    "mode",
                ):
                    for col in cleaned_data.columns:
                        if cleaned_data[col].isnull().any():
                            # For numeric columns
                            if pd.api.types.is_numeric_dtype(cleaned_data[col]):
                                if self.fill_value == "mean":
                                    fill_val = cleaned_data[col].mean()
                                    cleaned_data[col].fillna(fill_val, inplace=True)
                                elif self.fill_value == "median":
                                    fill_val = cleaned_data[col].median()
                                    cleaned_data[col].fillna(fill_val, inplace=True)
                                elif self.fill_value == "mode":
                                    fill_val = cleaned_data[col].mode()[0]
                                    cleaned_data[col].fillna(fill_val, inplace=True)
                            # For non-numeric (categorical) columns, only mode is generally applicable
                            else:
                                if self.fill_value == "mode":
                                    fill_val = cleaned_data[col].mode()[0]
                                    cleaned_data[col].fillna(fill_val, inplace=True)
                else:
                    # If fill_value is a constant (e.g., 0, "Unknown")
                    cleaned_data.fillna(self.fill_value, inplace=True)

        logger.info("Data cleaning finished.")
        return cleaned_data
