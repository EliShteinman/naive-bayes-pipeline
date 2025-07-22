# backend/nb_classifier/data_splitter.py
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from .logger_config import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """
    Splits a DataFrame into training and testing sets and ensures that
    the test set contains only values seen during training.
    """

    def __init__(self, data: pd.DataFrame, target_col: str):
        """
        Initializes the DataSplitter.

        Args:
            data (pd.DataFrame): The DataFrame to be split.
            target_col (str): The name of the target variable column.
        """
        self.data = data
        self.target_col = target_col

    def split_data(
        self, test_size: float = 0.3, random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the data into training and testing DataFrames.

        This method first splits the data stratifying by the target column.
        Then, it validates the test set to remove any rows containing feature
        values not present in the training set.

        Args:
            test_size (float): The proportion of the dataset to allocate to the test split.
            random_state (int): Seed for reproducibility.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing DataFrames.
        """
        logger.info(
            f"Splitting data with test_size={test_size} and random_state={random_state}."
        )

        # 1. Perform stratified split
        train_df, test_df = train_test_split(
            self.data,
            test_size=test_size,
            random_state=random_state,
            stratify=self.data[self.target_col],
        )

        logger.info(
            f"Initial split complete. Train set: {train_df.shape[0]} rows, Test set: {test_df.shape[0]} rows."
        )

        # 2. Ensure test set contains only known values
        feature_cols = [col for col in self.data.columns if col != self.target_col]
        train_uniques = {col: set(train_df[col].unique()) for col in feature_cols}

        # Create a boolean mask for rows in the test set that are valid
        # A row is valid if all its feature values have been seen in the training set
        valid_rows_mask = (
            test_df[feature_cols]
            .apply(lambda col: col.isin(train_uniques[col.name]))
            .all(axis=1)
        )

        # Identify and drop rows that are not valid
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

        # Reset index to ensure clean, continuous indices
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        logger.info("Data splitting and validation finished successfully.")
        return train_df, test_df
