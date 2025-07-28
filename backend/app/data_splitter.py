# app/data_splitter.py
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.app.logger_config import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """
    Splits a DataFrame into training and testing sets based on configuration
    provided at initialization. It can also optionally validate the test set
    to ensure it contains only values seen during training.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str,
        test_size: float = 0.3,
        random_state: int = 42,
        validate_test_set: bool = True,
    ):
        """
        Initializes the DataSplitter with the data and splitting configuration.

        Args:
            data (pd.DataFrame): The DataFrame to be split.
            target_col (str): The name of the target variable column.
            test_size (float): The proportion of the dataset to allocate to the test split.
            random_state (int): Seed for reproducibility.
            validate_test_set (bool): If True, removes rows from the test set that
                                      contain feature values not present in the training set.
        """
        self.data = data
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.validate_test_set = validate_test_set
        logger.info(
            f"DataSplitter initialized with test_size={self.test_size}, "
            f"random_state={self.random_state}, "
            f"validate_test_set={self.validate_test_set}"
        )

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing DataFrames using the
        configuration stored in the instance.
        """
        logger.info("Splitting data...")

        # 1. Perform stratified split using instance attributes
        train_df, test_df = train_test_split(
            self.data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.data[self.target_col],
        )

        logger.info(
            f"Initial split complete. Train set: {train_df.shape[0]} rows, Test set: {test_df.shape[0]} rows."
        )

        # 2. Optionally, validate the test set to ensure it contains only known values
        # <<< התניית הלוגיקה בפרמטר החדש >>>
        if self.validate_test_set:
            logger.info(
                "Validating test set to remove rows with unseen feature values."
            )

            feature_cols = [col for col in self.data.columns if col != self.target_col]
            train_uniques = {col: set(train_df[col].unique()) for col in feature_cols}

            valid_rows_mask = (
                test_df[feature_cols]
                .apply(lambda col: col.isin(train_uniques[col.name]))
                .all(axis=1)
            )

            indices_to_drop = test_df[~valid_rows_mask].index

            if not indices_to_drop.empty:
                num_dropped = len(indices_to_drop)
                logger.warning(
                    f"Dropping {num_dropped} rows from the test set because they "
                    "contain feature values not present in the training set."
                )
                test_df = test_df.drop(index=indices_to_drop)
                logger.info(
                    f"Test set size after dropping unseen values: {test_df.shape[0]} rows."
                )
            else:
                logger.info("Test set validation passed. No rows were dropped.")
        else:
            logger.info("Skipping test set validation as per configuration.")

        # Reset index to ensure clean, continuous indices
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        logger.info("Data splitting and validation finished successfully.")
        return train_df, test_df
