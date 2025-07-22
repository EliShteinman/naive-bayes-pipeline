# nb_classifier/data_handler.py
from typing import Any, Dict, Hashable, List, Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from nb_classifier.logger_config import get_logger

logger = get_logger(__name__)


class DataHandler:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_clean_data_as_dicts(self) -> list[dict[Hashable, Any]]:
        """
        Endpoint to get the entire cleaned dataset as a list of dictionaries.

        Returns:
            List[Dict[str, Any]]: The data, ready to be sent as a JSON array.
        """
        df = self._load_and_clean_df()
        return df.to_dict(orient="records")

    def get_split_data_as_dicts(
        self, target_col: str, test_size: float = 0.3, random_state: int = 42
    ) -> tuple[DataFrame, DataFrame]:
        full_df = self._load_and_clean_df()

        # הלוגיקה של split_and_validate נמצאת כאן
        train_df, test_df = self._split_and_validate_df(
            full_df, target_col, test_size, random_state
        )

        return train_df, test_df

    # מתודות עזר פנימיות שמטפלות ב-DataFrames
    def _load_and_clean_df(self) -> pd.DataFrame:
        """
        Loads data from the CSV file, removes known problematic columns,
        and filters out columns with no variance.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        print(self.data_path)
        cleaned_data = (
            pd.read_csv(self.data_path)
            .drop(columns=["stalk-root"], errors="ignore")
            .loc[:, lambda df: df.nunique() > 1]
        )
        return cleaned_data

    def _split_and_validate_df(
        self,
        data: pd.DataFrame,
        target_col: str,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data into training and testing sets and validates the test set.

        It ensures that the test set only contains values that were also present
        in the training set, preventing errors in models that can't handle
        unseen categories. This is done efficiently using vectorized operations.

        Args:
            data (pd.DataFrame): The full dataset to split.
            target_col (str): The name of the target column for stratification.
            test_size (float): The proportion of the data to use for the test set.
            random_state (int): The seed for reproducibility.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple of (train_df, validated_test_df).
        """
        train_df, test_df = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=data[target_col],
        )

        feature_cols = [col for col in data.columns if col != target_col]
        train_uniques = {col: set(train_df[col].unique()) for col in feature_cols}

        mask_to_drop = (
            ~test_df[feature_cols]
            .apply(lambda col: col.isin(train_uniques[col.name]))
            .all(axis=1)
        )

        indices_to_drop = test_df[mask_to_drop].index

        if not indices_to_drop.empty:
            print(
                f"הוסרו {len(indices_to_drop)} שורות מסט הבדיקה בגלל ערכים שלא הופיעו באימון."
            )
            test_df = test_df.drop(index=indices_to_drop)

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    @staticmethod
    def get_data_as_list_of_dicts(data: pd.DataFrame) -> List[Dict[Hashable, Any]]:

        return data.to_dict(orient="records")
