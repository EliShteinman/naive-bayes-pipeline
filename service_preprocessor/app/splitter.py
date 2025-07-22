# service_training/app/preprocessing/splitter.py
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .common.logger_config import get_logger

logger = get_logger(__name__)


class Splitter:
    """
        Splits a DataFrame into training and testing sets.

        This class wraps scikit-learn's `train_test_split` function, providing
        options for stratified splitting. It also includes a crucial post-split
        check to ensure that the test set does not contain categorical values
        that were not present in the training set, which is vital for many
    -   models.
    """

    def __init__(
        self,
        test_size: float = 0.3,
        random_state: int = 42,
        stratify: bool = True,
    ):
        """
        Initializes the Splitter with configuration for the split.

        Args:
            test_size (float, optional): The proportion of the dataset to allocate
                                         to the test set. Defaults to 0.3 (30%).
            random_state (int, optional): A seed for the random number generator
                                          to ensure reproducible splits. Defaults to 42.
            stratify (bool, optional): If True, performs stratified sampling based on
                                       the target column to maintain the same class
                                       distribution in both train and test sets.
                                       Defaults to True.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        logger.info(
            f"Splitter initialized with test_size={test_size}, "
            f"random_state={random_state}, stratify={stratify}."
        )

    def split(
        self, data: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs the train-test split and cleans the test set.

        First, it splits the data into training and testing sets using the
        configured parameters. Afterwards, it performs a validation step: it
        checks each feature column to see if the test set contains categorical
        values that do not exist in the training set. If such values are found,
        it logs a warning and removes the corresponding rows from the test set
        to prevent errors during model prediction.

        Args:
            data (pd.DataFrame): The full DataFrame to be split.
            target_col (str): The name of the target column, used for stratification.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training
                                               DataFrame and the cleaned testing
                                               DataFrame.
        """
        logger.info(f"Splitting data with shape {data.shape} on target '{target_col}'.")

        # Determine if stratification should be used
        stratify_values = data[target_col] if self.stratify else None

        train_df, test_df = train_test_split(
            data,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify_values,
        )

        logger.info(
            f"Initial split complete. Train set: {train_df.shape[0]} rows, Test set: {test_df.shape[0]} rows."
        )

        # --- Post-split validation for categorical values ---
        feature_cols = [col for col in data.columns if col != target_col]

        for col in feature_cols:
            train_values = set(train_df[col].unique())
            test_values = set(test_df[col].unique())

            # Find values that are in the test set but not in the train set
            unseen_values = test_values - train_values

            if unseen_values:
                logger.warning(
                    f"Column '{col}' has values in the test set that are not in the train set: {unseen_values}. "
                    f"Rows with these values will be removed from the test set."
                )
                initial_test_rows = test_df.shape[0]
                # Filter out rows from the test set that contain the unseen values
                test_df = test_df[~test_df[col].isin(unseen_values)]
                logger.info(
                    f"Removed {initial_test_rows - test_df.shape[0]} rows from the test set for column '{col}'."
                )

        logger.info(
            f"Final split sizes. Train set: {train_df.shape[0]} rows, Test set: {test_df.shape[0]} rows."
        )
        return train_df, test_df
